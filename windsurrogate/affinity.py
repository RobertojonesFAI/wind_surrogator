"""Pin serial solver runs to physical cores and to their local memory (NUMA node).

On multi-socket nodes, dozens of independent OpenFOAM runs left to the scheduler get
moved between sockets and the kernel's automatic NUMA balancing keeps migrating their
memory, which costs a lot of system time on memory-bound solvers. Binding each run to
one physical core and to the memory of that core's socket avoids both.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def _read_int(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def _numa_node(cpu: int) -> int:
    nodes = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    return int(nodes[0].name[4:]) if nodes else 0


def core_slots() -> list[tuple[int, int]]:
    """One (logical cpu, numa node) per physical core this process may use.

    Hyper-thread siblings are dropped, and the list alternates between NUMA nodes so that
    taking the first k slots spreads k runs evenly over the sockets.
    """
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:          # not Linux
        return []
    seen, per_node = set(), {}
    for cpu in allowed:
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (_read_int(topo / "physical_package_id"), _read_int(topo / "core_id", cpu))
        if key in seen:
            continue
        seen.add(key)
        per_node.setdefault(_numa_node(cpu), []).append(cpu)
    slots, lists = [], [per_node[n] for n in sorted(per_node)]
    for i in range(max((len(x) for x in lists), default=0)):
        for node, cpus in zip(sorted(per_node), lists):
            if i < len(cpus):
                slots.append((cpus[i], node))
    return slots


def pin_prefix(cpu: int, node: int) -> str:
    """Shell prefix that runs a command on `cpu` with memory bound to `node`."""
    if shutil.which("numactl"):
        return f"numactl --physcpubind={cpu} --membind={node} "
    if shutil.which("taskset"):
        return f"taskset -c {cpu} "
    return ""


def pinning_enabled(cfg, concurrent: int) -> bool:
    mode = str(cfg["solver"].get("pin_cores", "auto")).lower()
    if mode in ("false", "no", "off", "0"):
        return False
    if mode in ("true", "yes", "on", "1"):
        return True
    return concurrent > 1 and len(core_slots()) >= concurrent


def describe() -> str:
    slots = core_slots()
    nodes = sorted({n for _, n in slots})
    tool = "numactl" if shutil.which("numactl") else "taskset" if shutil.which("taskset") else "none"
    return f"{len(slots)} physical cores on NUMA node(s) {nodes}, pinning tool: {tool}"
