"""Keyhole-depth oscillation spectra of keyhole runs (keyhole_ti64.py, keyhole_overhang.py).

X-ray studies of keyholes report depth oscillations in the 1-10 kHz range in the unstable
regime, and pores pinching off the keyhole tip. Per run and window this prints and stores:
the mean and rms depth, the dominant frequency, the share of the fluctuation power in the bands
below 1 kHz, 1-10 kHz, 10-50 kHz and above 50 kHz, and the rate at which gas pockets pinch off.

Pinch-offs: the gas pockets in the log (every 1 us) are followed from one entry to the next by
their centroids. A pinch-off is a pocket of at least 10 um (equivalent diameter), living at
least 5 us, whose walls reached the solidus (the cold gaps between powder particles and
one-cell specks are not counted). ``bubbles_formed`` in the log counts every closed gas region
the labelling ever saw, specks and powder voids included, and is not a pinch-off rate.

Windows (the beam speed from the logged laser position):
  * steady:  laser on, beam at >= 90 % of its top speed, in the second half of that stretch of
             track (as the depth statistics of keyhole_figure.py);
  * turn:    turn-point runs only: the beam below 90 % of its top speed (deceleration, reversal
             and acceleration, where the keyhole deepens and the turn-point pores pinch off,
             some of them only while the beam speeds up again); the speed changes through it,
             so the fluctuations about a 100 us Gaussian running mean are analysed (the slow
             deepening and recovery removed; above ~3 kHz kept), peak searched above 2 kHz.

The frequency resolution is 1 / (window length): 4 kHz for a 250 us window. Resolving the
1-10 kHz band properly needs millisecond-long tracks.

    python tools/lbm/keyhole_spectrum.py RUN.json [RUN.json ...] [--out docs/img/keyhole_spectrum.png]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from tpmsfem.process.keyhole import depth_spectrum  # noqa: E402


def windows(d):
    L = d["log"]
    t = np.array([r["t"] for r in L])
    on = np.array([bool(r.get("laser_on", r.get("x_l") is not None)) for r in L])
    x = np.array([r["x_l"] if r.get("x_l") is not None else np.nan for r in L])
    v = np.abs(np.gradient(x, t))
    v[~on] = 0.0
    vmax = np.nanmax(v[on]) if on.any() else 0.0
    out = {}
    fast = on & (v >= 0.9 * vmax)
    if fast.any():
        # the longest stretch at top speed, its second half
        idx = np.flatnonzero(fast)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        r = max(runs, key=len)
        a, b = r[0] + len(r) // 2, r[-1]
        out["steady"] = (t[a], t[b])
    if d.get("scenario", "").startswith("turn"):
        slow = on & (v < 0.9 * vmax)
        if slow.any():
            idx = np.flatnonzero(slow)
            out["turn"] = (t[idx[0]], t[idx[-1]])
    return out


def pinch_offs(L, t0, t1, solidus, d_min=10.0, life_min=5e-6):
    """Pockets that pinched off in [t0, t1] (see the module docstring): their count, and how
    many of them lived 50 us or more."""
    tracks, live = [], []
    for r in L:
        t = r["t"]
        if t < t0 or t > t1:
            continue
        cur = []
        for b in r.get("bubbles", []):
            c = np.array(b["centroid_um"], float)
            de = (6.0 * b["volume_um3"] / np.pi) ** (1.0 / 3.0)
            cur.append((c, de, b.get("T_wall_max", 0.0) or 0.0))
        used, new_live = set(), []
        for c, de, Tw in cur:
            best, bd = None, None
            for k in live:
                if k in used:
                    continue
                dist = np.linalg.norm(tracks[k]["c"] - c)
                if dist <= max(8.0, max(de, tracks[k]["d"])) and (bd is None or dist < bd):
                    best, bd = k, dist
            if best is None:
                tracks.append(dict(c=c, d=de, dmax=de, Tw=Tw, t0=t, t1=t))
                best = len(tracks) - 1
            else:
                tr = tracks[best]
                tr.update(c=c, d=de, dmax=max(tr["dmax"], de), Tw=max(tr["Tw"], Tw), t1=t)
            used.add(best)
            new_live.append(best)
        live = new_live
    ok = [tr for tr in tracks if tr["dmax"] >= d_min and tr["t1"] - tr["t0"] >= life_min and tr["Tw"] >= solidus]
    return len(ok), sum(1 for tr in ok if tr["t1"] - tr["t0"] >= 50e-6)


def analyse(path):
    d = json.load(open(path))
    L = d["log"]
    t = np.array([r["t"] for r in L])
    dep = np.array([r["depth"] if r.get("depth") is not None else np.nan for r in L])
    solidus = 1250.0 if "slab" in d else 1604.0           # CoCrFeMnNi overhangs, Ti-6Al-4V
    res = {}
    for name, (t0, t1) in windows(d).items():
        if t1 - t0 < 40e-6:
            continue
        s = depth_spectrum(t, dep, t0, t1, **(dict(smooth=100e-6, f_min=2e3) if name == "turn" else {}))
        n_po, n_long = pinch_offs(L, t0, t1, solidus)
        rate = n_po / ((t1 - t0) * 1e3)
        res[name] = dict(pinch_offs=n_po, pinch_offs_lived_50us=n_long, t0_us=t0 * 1e6, t1_us=t1 * 1e6, T_us=s["T"] * 1e6, df_kHz=s["df"] / 1e3,
                         mean_um=s["mean"] * 1e6, rms_um=s["rms"] * 1e6, f_peak_kHz=s["f_peak"] / 1e3,
                         band_fraction=s["band_fraction"], pinch_off_per_ms=rate,
                         f_kHz=(s["f"] / 1e3).tolist(), psd_um2_per_kHz=(s["psd"] * 1e12 * 1e3).tolist())
    return d, res


def main(paths, out=None):
    rows = {}
    for p in paths:
        d, res = analyse(p)
        tag = os.path.basename(p)[:-5]
        rows[tag] = dict(P=d.get("P"), v=d.get("v"), scenario=d.get("scenario"), dtype=d.get("dtype", "float64"),
                         windows=res)
        for name, r in res.items():
            b = r["band_fraction"]
            print(f"{tag:32s} {name:6s} {r['t0_us']:6.0f}-{r['t1_us']:6.0f} us (df {r['df_kHz']:4.1f} kHz): "
                  f"depth {r['mean_um']:6.1f} ± {r['rms_um']:5.1f} um, peak {r['f_peak_kHz']:6.1f} kHz, "
                  f"power <1k {b['<1kHz']:.2f} | 1-10k {b['1-10kHz']:.2f} | 10-50k {b['10-50kHz']:.2f} | "
                  f">50k {b['>50kHz']:.2f}; pinch-offs {r['pinch_offs']} ({r['pinch_off_per_ms']:.0f}/ms, "
                  f"{r['pinch_offs_lived_50us']} lived >= 50 us)")
    js = (out or os.path.join(HERE, "keyhole_runs", "keyhole_spectrum.png"))[:-4] + ".json"
    json.dump({k: dict(v, windows={w: {kk: vv for kk, vv in r.items() if kk not in ("f_kHz", "psd_um2_per_kHz")}
                                   for w, r in v["windows"].items()}) for k, v in rows.items()},
              open(js, "w"), indent=1)
    print("wrote", js)
    if out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        short = {tag: tag.replace("_dx3.33um", "").replace("_f32", "").replace("overhang_", "") for tag in rows}
        cols = dict(zip(rows, ["#2a78d6", "#eb6834", "#1f9e74", "#8a4fd0", "#c2410c", "#6b7280", "#0b3a75", "#9ec5ee"]))
        fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
        for row, name, title in ((0, "steady", "(a) steady window (beam at full speed)"),
                                 (1, "turn", "(b) around the turn: fluctuations about a 100 µs running mean")):
            ax = fig.add_subplot(gs[row, 0])
            for tag, v in rows.items():
                r = v["windows"].get(name)
                if r is None:
                    continue
                f, p = np.array(r["f_kHz"]), np.array(r["psd_um2_per_kHz"])
                ax.loglog(f[1:], p[1:], color=cols[tag], lw=1.1,
                          label=f"{short[tag]}: peak {r['f_peak_kHz']:.0f} kHz, rms {r['rms_um']:.0f} µm, "
                                f"{r['pinch_offs']} pinch-offs")
            ax.axvspan(1, 10, color="#e4e3df", alpha=0.7, lw=0, zorder=0)
            ax.text(3.16, 0.97, "1-10 kHz", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                    color="#555")
            ax.set_xlim(0.5, 600)
            ax.set_ylabel("depth PSD [µm² / kHz]")
            ax.set_title(title, loc="left", fontsize=9)
            ax.grid(True, which="major", color="#eee", lw=0.5)
            ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax.set_xlabel("frequency [kHz]")
        ax = fig.add_subplot(gs[:, 1])
        labels, data = [], []
        for tag, v in rows.items():
            for name, r in v["windows"].items():
                labels.append(f"{short[tag]}, {name}")
                data.append([r["band_fraction"][b] for b in ("<1kHz", "1-10kHz", "10-50kHz", ">50kHz")])
        data = np.array(data)
        left = np.zeros(len(labels))
        for j, (b, c) in enumerate(zip(("< 1 kHz", "1-10 kHz", "10-50 kHz", "> 50 kHz"),
                                       ("#9ec5ee", "#2a78d6", "#eb6834", "#6b7280"))):
            ax.barh(range(len(labels)), data[:, j], left=left, color=c, label=b, height=0.7)
            left += data[:, j]
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the depth-fluctuation power")
        ax.set_title("(c) where the fluctuation power is", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.07))
        fig.suptitle("Keyhole depth oscillations, 3.33 µm cells (depth logged every 1 µs)", x=0.01, ha="left",
                     fontsize=10)
        fig.savefig(out, dpi=130)
        print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(1)
    main(args, out)
