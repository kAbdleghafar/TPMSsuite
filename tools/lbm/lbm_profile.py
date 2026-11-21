"""Where does a free-surface keyhole step spend its time? Wraps the parts of
``FreeSurfaceMeltPool.run`` with timers (synchronising the GPU around each), runs a few hundred
steps and prints the breakdown and the speed (MLUPS). Used by ``--profile N`` of
keyhole_ti64.py and keyhole_overhang.py.

Parts timed: the lattice step (fused CUDA / numba / array), the heat step, the surface terms
(Marangoni, recoil, evaporation, curvature), the ray tracer (host numba + transfers), the
bubble labelling, the solid/liquid coupling and the new-cell enthalpy. "other" is the rest of
the loop (Python, diagnostics).
"""
import time

import numpy as np

from tpmsfem.backend import get_xp


class _Stop(Exception):
    pass


def _sync():
    xp = get_xp()
    if xp is not np:
        xp.cuda.Stream.null.synchronize()


def profile(md, laser, path, n_steps=300, warmup=30):
    """Run ``warmup + n_steps`` steps of ``md`` along ``path`` and return the time per step
    [ms] of each part, the total, and MLUPS (over the timed steps only)."""
    acc = {}
    state = dict(on=False)

    def wrap(obj, name, label):
        f = getattr(obj, name)

        def timed(*a, **k):
            if not state["on"]:
                return f(*a, **k)
            _sync()
            t0 = time.perf_counter()
            r = f(*a, **k)
            _sync()
            acc[label] = acc.get(label, 0.0) + time.perf_counter() - t0
            return r
        setattr(obj, name, timed)

    wrap(md.lbm, "step", "lattice step")
    wrap(md, "_heat_step", "heat")
    wrap(md, "_surface_terms", "surface terms")
    wrap(md, "_ray_source", "ray tracer")
    wrap(md, "_couple", "solid/liquid coupling")
    if hasattr(md, "_new_metal_cells"):
        wrap(md, "_new_metal_cells", "new cells")
    if md.bubbles is not None:
        wrap(md.bubbles, "update", "bubbles")
    count = dict(n=0, t0=None)

    def cb(m, t, x, y):
        count["n"] += 1
        if count["n"] == warmup:
            _sync()
            state["on"] = True
            count["t0"] = time.perf_counter()
        if count["n"] >= warmup + n_steps:
            _sync()
            raise _Stop()

    try:
        md.run(laser, path, callback=cb, callback_every=1)
    except _Stop:
        pass
    total = time.perf_counter() - count["t0"]
    n = count["n"] - warmup
    rows = {k: 1e3 * v / n for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
    rows["other"] = 1e3 * total / n - sum(rows.values())
    cells = float(np.prod(md.shape))
    return dict(ms_per_step=rows, total_ms=1e3 * total / n, mlups=cells * n / total / 1e6, steps=n,
                cells=int(cells), dtype=np.dtype(md.dtype).name, kernel=md.lbm.kernel)


def report(r):
    print(f"{r['steps']} steps, {r['cells'] / 1e6:.2f} M cells, {r['dtype']}, kernel {r['kernel']}: "
          f"{r['total_ms']:.1f} ms per step, {r['mlups']:.1f} MLUPS", flush=True)
    for k, v in r["ms_per_step"].items():
        print(f"  {k:22s} {v:8.2f} ms  {100 * v / r['total_ms']:5.1f} %", flush=True)
