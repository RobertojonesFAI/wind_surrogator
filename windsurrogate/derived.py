"""Derived wind products for downstream models (e.g. a thermal model)."""
from __future__ import annotations

import numpy as np


def derived_fields(u_e, v_n, w, k, u_along, v_cross, U_in, dx, shadow_thr=0.6, compress_thr=1.2):
    """All inputs (H, rows, cols) except U_in (H,). Returns dict of (H, rows, cols) arrays.

    u_e, v_n   east/north horizontal components (m/s)
    u_along    component along the free-stream direction (negative = reversed flow)
    v_cross    component to the left of the free-stream direction
    """
    Uin = np.asarray(U_in, np.float32)[:, None, None]
    speed = np.hypot(u_e, v_n)
    speedup = speed / Uin
    dir_from = (np.degrees(np.arctan2(-u_e, -v_n)) + 360.0) % 360.0
    deflection = np.degrees(np.arctan2(v_cross, u_along))           # + = turned to the left
    ti = np.sqrt(np.maximum(2.0 * k / 3.0, 0)) / np.maximum(np.sqrt(speed**2 + w**2), 0.1)
    # Horizontal convergence on terrain-following surfaces (1/s): > 0 where air piles up.
    dudx = np.gradient(u_e, dx, axis=-1)
    dvdy = -np.gradient(v_n, dx, axis=-2)    # rows go north -> south
    convergence = -(dudx + dvdy)
    return dict(
        speed=speed, speedup=speedup, dir_from=dir_from, deflection=deflection, w=w, tke=k, ti=ti,
        convergence=convergence,
        recirculation=(u_along < 0).astype(np.float32),
        shadow=(speedup < shadow_thr).astype(np.float32),
        compression=(speedup > compress_thr).astype(np.float32),
    )
