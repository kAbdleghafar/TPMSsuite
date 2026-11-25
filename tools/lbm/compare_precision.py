"""Is float32 good enough? Compares keyhole runs made in float64 and float32 (same case, same
command apart from ``--dtype float32``): the quantities the closed-keyhole study reports, side by
side, with a verdict, and a figure of the depth over time.

A keyhole is chaotic: two runs that differ in round-off follow different trajectories after a
while, so the time series are not compared point by point. The statistics are: the steady depth
(mean and spread over the second half of the track), the absorbed fraction, the depth when the
power reaches zero and after the turn-off, the end crater, the bubble counts and pores, and mass
and energy conservation. Tolerances (a float32 run passes if all hold):

* steady depth: means within 5 %;
* depth after the turn-off (+10, +30, +100 us) and end crater: within 10 um or 15 %;
* absorbed fraction: within 0.03;
* pores at the end: the same number;
* mass error below 1e-5, booked energy residual below 1 %.

    python tools/lbm/compare_precision.py tools/lbm/keyhole_runs/T3_dx3.33um.json tools/lbm/keyhole_runs/T3_dx3.33um_f32.json
    python tools/lbm/compare_precision.py --all tools/lbm/keyhole_runs        # every *_f32.json with its float64 twin
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def stats(path):
    d = json.load(open(path))
    L = d["log"]
    t = np.array([r["t"] for r in L])
    on = np.array([r.get("laser_on", r.get("x_l") is not None) for r in L])
    dep = np.array([r.get("depth", np.nan) for r in L]) * 1e6
    ab = np.array([r.get("ray_absorbed", np.nan) for r in L])
    x = np.array([r["x_l"] if r.get("x_l") is not None else np.nan for r in L])
    out = dict(path=path, dtype=d.get("dtype", "float64"), finished=d.get("finished", False))
    if on.any():
        xs = x[on]
        steady = on & (x >= np.nanmin(xs) + 0.5 * (np.nanmax(xs) - np.nanmin(xs)))
        out.update(steady_depth=float(np.nanmean(dep[steady])), steady_std=float(np.nanstd(dep[steady])),
                   absorbed=float(np.nanmean(ab[steady])))
        if (~on).any() and on[0]:
            k = int(np.argmax(~on))
            t_off = t[k]
            out["depth_at_off"] = float(dep[k - 1])
            for dt_us in (10, 30, 100):
                j = int(np.argmin(np.abs(t - (t_off + dt_us * 1e-6))))
                if abs(t[j] - t_off - dt_us * 1e-6) < 2e-6:
                    out[f"depth_off+{dt_us}us"] = float(dep[j])
    out["crater"] = float(dep[-1])
    last = L[-1]
    out["bubbles_formed"] = last.get("bubbles_formed", last.get("formed"))
    out["bubbles_released"] = last.get("bubbles_released", last.get("released"))
    out["pores"] = sum(1 for b in last.get("bubbles", []) if b.get("pore"))
    for k in ("mass_error", "energy_residual", "seconds", "mlups", "steps"):
        out[k] = d.get(k)
    out["t_us"] = (t * 1e6).tolist()
    out["depth_series"] = dep.tolist()
    return out


def verdict(a, b):
    """Rows (quantity, float64, float32, ok) and the overall pass."""
    rows = []

    def add(name, va, vb, ok, fmt="{:.1f}"):
        f = (lambda v: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v))
        rows.append((name, f(va), f(vb), ok))

    if "steady_depth" in a and "steady_depth" in b:
        add("steady depth [um]", a["steady_depth"], b["steady_depth"],
            abs(b["steady_depth"] - a["steady_depth"]) <= 0.05 * a["steady_depth"])
        add("  spread (std) [um]", a["steady_std"], b["steady_std"], None)
        add("absorbed fraction", a["absorbed"], b["absorbed"], abs(b["absorbed"] - a["absorbed"]) <= 0.03, "{:.3f}")
    for k in ("depth_at_off", "depth_off+10us", "depth_off+30us", "depth_off+100us", "crater"):
        if k in a and k in b:
            tol = max(10.0, 0.15 * abs(a[k]))
            add(f"{k.replace('_', ' ')} [um]", a[k], b[k], abs(b[k] - a[k]) <= tol)
    add("bubbles formed / released", None, None, None)
    rows[-1] = ("bubbles formed / released", f"{a['bubbles_formed']} / {a['bubbles_released']}",
                f"{b['bubbles_formed']} / {b['bubbles_released']}", None)
    add("pores at the end", a["pores"], b["pores"], a["pores"] == b["pores"], "{:d}")
    add("mass error", a["mass_error"], b["mass_error"], b["mass_error"] is not None and abs(b["mass_error"]) < 1e-5, "{:.1e}")
    add("energy residual", a["energy_residual"], b["energy_residual"],
        b["energy_residual"] is not None and abs(b["energy_residual"]) < 0.01, "{:.2%}")
    add("wall time [min]", (a["seconds"] or np.nan) / 60, (b["seconds"] or np.nan) / 60, None)
    add("MLUPS", a["mlups"], b["mlups"], None)
    ok = all(r[3] for r in rows if r[3] is not None)
    return rows, ok


def compare(fa, fb, fig=True):
    a, b = stats(fa), stats(fb)
    rows, ok = verdict(a, b)
    print(f"\n{os.path.basename(fa)}  vs  {os.path.basename(fb)}")
    print(f"  {'quantity':30s} {'float64':>16s} {'float32':>16s}  check")
    for name, va, vb, good in rows:
        print(f"  {name:30s} {va:>16s} {vb:>16s}  {'' if good is None else ('ok' if good else 'FAIL')}")
    if a["seconds"] and b["seconds"]:
        print(f"  speed-up: {a['seconds'] / b['seconds']:.1f}x")
    print(f"  VERDICT: {'float32 reproduces float64 within the tolerances' if ok else 'float32 does NOT pass'}")
    if not (a["finished"] and b["finished"]):
        print("  (at least one run is not finished: the comparison is partial)")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(a["t_us"], a["depth_series"], color="#2a78d6", lw=1.2, label=f"float64 ({(a['seconds'] or 0) / 60:.0f} min)")
        ax.plot(b["t_us"], b["depth_series"], color="#eb6834", lw=1.0, label=f"float32 ({(b['seconds'] or 0) / 60:.0f} min)")
        ax.set_xlabel("time [us]")
        ax.set_ylabel("keyhole depth [um]")
        ax.set_title(f"{os.path.basename(fa).replace('.json', '')}: {'PASS' if ok else 'FAIL'}", loc="left", fontsize=10)
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.legend(frameon=False, fontsize=8)
        out = fb.replace(".json", "_precision.png")
        f.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  figure: {out}")
    return ok


def main():
    args = sys.argv[1:]
    if args and args[0] == "--all":
        folder = args[1] if len(args) > 1 else os.path.join(HERE, "keyhole_runs")
        pairs = [(p.replace("_f32.json", ".json"), p) for p in sorted(glob.glob(os.path.join(folder, "*_f32.json")))]
        pairs = [(a, b) for a, b in pairs if os.path.exists(a)]
        if not pairs:
            print("no float32 runs with a float64 twin found in", folder)
            return 1
    elif len(args) == 2:
        pairs = [tuple(args)]
    else:
        print(__doc__)
        return 1
    results = [compare(a, b) for a, b in pairs]
    return 0 if all(results) else 2


if __name__ == "__main__":
    sys.exit(main())
