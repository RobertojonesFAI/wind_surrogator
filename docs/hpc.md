# Clusters and cloud VMs

The CFD cases are independent serial jobs, so the dataset scales almost linearly with the number of cores.

## Environment

```bash
bash hpc/setup_env.sh                                   # micromamba + env "foam" in $HOME/mamba
MAMBA_ROOT_PREFIX=/scratch/$USER/mamba bash hpc/setup_env.sh   # if $HOME quota is small
```

The env contains OpenFOAM v2412 (conda-forge) and the Python packages for the CFD side. Training needs PyTorch and a GPU; it is usually simpler on Colab or on a GPU node: `pip install -e ".[ml]"`.

## Slurm job array

```bash
cd wind_surrogator
mkdir -p logs
PROJECT=/scratch/$USER/windsurrogate_runs/idaho NUM_SHARDS=8 \
  sbatch -p <partition> --array=0-7 hpc/run_cfd.slurm
```

- **One array task is one shard on one whole node.** It runs one case per physical core; hyper-threading is detected and not used.
- **Copy the project first.** `PROJECT` must contain the same `cases.jsonl` and `dem/` as the main project, plus `dataset/` if you want already-finished cases skipped.
- **Stop time.** `STOP_HOURS` defaults to the job's remaining time minus 2 hours, so no new case starts too close to the time limit.
- **Resubmitting.** Submitting again with the same `NUM_SHARDS` continues where it stopped.
- **Preemptible partitions.** Add `--requeue`. A preempted task only loses the cases that were running.
- **Limits.** Check per-user node limits and maximum wall time:

  ```bash
  scontrol show partition | grep -E "PartitionName|MaxNodes|MaxTime"
  sacctmgr show qos format=Name,MaxTRESPU%30,MaxJobsPU,MaxWall
  ```
- **Default wall time.** Always set the time limit explicitly (`-t`); some clusters default to a few hours.

### Sizing

At about 1.0–1.5 cases per core-hour with the v1 mesh:

| Nodes × cores | Days | Core-hours | Cases (about 80 % efficiency) |
|---|---|---|---|
| 1 × 48 | 1 | 1,150 | ~900–1,400 |
| 8 × 48 | 7 | 64,500 | ~50,000–80,000 |
| 8 × 48 | 14 | 129,000 | ~100,000–160,000 |

The efficiency factor covers queue gaps, cases cut by the time limit, and cases that run to 1,500 iterations.

Each case writes ~1.4 MB, so 100,000 cases need ~140 GB. Running cases also need ~0.3 GB of scratch each, preferably on node-local disk.

More cases of the same two regions add little once the terrain is well covered. A large allocation is better spent on new regions or on the stratified v2 ([v2-stratified-design.md](v2-stratified-design.md)).

## Single VM (e.g. a cloud VM)

```bash
bash hpc/setup_env.sh
PROJECT=$HOME/runs/idaho bash hpc/run_vm.sh          # prepare if needed, then run detached
tail -f ~/windcfd.log
bash hpc/run_vm.sh status
```

- **Spot / preemptible VMs** work well: if the VM is stopped, start it again and re-run `run_vm.sh`.
- **Moving results:** copy the `.npz` files back to the main dataset folder, for example through a storage bucket. Copy only files that do not already exist there, so no finished case is overwritten.
