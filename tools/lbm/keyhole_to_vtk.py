"""Export a keyhole run (tools/lbm/keyhole_runs/<tag>.json, _sections.npz, _final.npz) to VTK
files for ParaView. No extra dependencies: VTK XML ImageData with raw appended data.

Writes, next to the run (or to ``--out``):
  <tag>_vtk/section_0000.vti ...   the centreline x-z section every saved step (``fill`` = metal
                                    fill level, ``T`` [degC]; field data ``time_us``,
                                    ``laser_on``, ``x_laser_um``, ``depth_um``, ``y_plane_um``)
  <tag>_section.pvd                 the time series of those sections (open this one)
  <tag>_section_return.pvd          turn-point runs: the same on the return track
  <tag>_3d.pvd                      runs made with --snapshots: the whole domain every few us
                                    (``fill``, ``T``, ``flag``; zlib-compressed files
                                    <tag>_vtk/snap_*.vti). For the cut view of a movie run
                                    tools/lbm/paraview_keyhole_3d.py in ParaView.
  <tag>_final.vti                   the 3-D state at the end of the run (``fill``, ``flag``
                                    0 gas / 1 interface / 2 liquid / 3 wall, ``T_peak`` [degC])

Coordinates in micrometres, z = 0 at the original plate surface (negative: into the plate).
All arrays are point data, one point per solver cell at its centre, so Contour and Threshold
work directly (no Cell Data to Point Data step).
  * sections (default): flat 2-D images, x along the track horizontally and the height z on the
    VTK y axis (ParaView labels it Y). With --xz they are written at their true place instead
    (the x-z plane at the track's y); then press the "-Y" camera button to look at them.
  * 3-D end state: x along the track, y across it, z up.

In ParaView:
  * section: File > Open <tag>_section.pvd, Apply. The image appears coloured by ``fill``;
    pick ``T`` in the colour menu for the temperature, press Play for the time series.
    Keyhole wall line: Filters > Contour, Contour By fill, value 0.5. If you only see a line
    or a box, the representation is not "Slice"/"Surface" or the camera looks edge-on: set
    the representation to Surface and press the "-Z" camera button (or "-Y" for --xz files).
  * 3-D end state: open <tag>_final.vti, Apply. ParaView first shows only its Outline (a box):
    add Filters > Contour, Contour By fill, value 0.5 (the surface, end crater and any pores), or
    Filters > Threshold on T_peak from the solidus up for the melted volume:
    1604 degC for Ti-6Al-4V (solidus 1877 K; liquidus 1660 degC), 1250 degC for CoCrFeMnNi
    (the overhang runs; liquidus 1340 degC).

Notes:
  * The outermost cells of the solver are walls (fill 0); they are written with the fill of their
    inner neighbour so that a fill = 0.5 contour shows only the metal surface.
  * In the 3-D snapshots ``T`` is extended two cells into the gas (mean of the metal
    neighbours) so that a clip through the surface is coloured; further out it is NaN.
  * ``T`` is NaN in empty gas cells (the solver has no temperature there): ParaView draws them in
    the NaN colour (grey by default), and a temperature contour stops at the free surface.
  * Temperatures and fill are stored as float16 by the run scripts: resolution 1 degC up to
    2048 degC, 2 degC up to 4096, 4 degC above.
  * The section is the plane of the first track (y in the run's .json). Turn-point runs made
    with the current keyhole_ti64.py also save the return track (y + hatch): written as
    <tag>_section_return.pvd (files return_0000.vti ...). Older turn runs only have the first track.

    python tools/lbm/keyhole_to_vtk.py tools/lbm/keyhole_runs/T3_dx3.33um.json
"""
import argparse
import json
import os
import sys

import numpy as np


def open_walls(fill):
    """The solver's boundary layer (the outermost cells) is wall, stored with fill 0, so a fill = 0.5
    contour would also trace the box. For display, the wall cells take the fill of their inner
    neighbour; the flag array of the 3-D end state still marks them (3)."""
    f = np.array(fill, dtype=np.float32)
    for ax in range(f.ndim):
        if f.shape[ax] < 3:
            continue
        sl = [slice(None)] * f.ndim
        for a, b in ((0, 1), (-1, -2)):
            sl[ax] = a
            src = list(sl); src[ax] = b
            f[tuple(sl)] = f[tuple(src)]
    return f


def extend_into_gas(T, layers=2):
    """NaN (gas) cells next to metal take the mean of their metal neighbours, ``layers`` cells
    deep. Only for the display of 3-D clips: a cut through the metal surface interpolates between
    a metal point and a gas point, and with NaN there it would be drawn in the NaN colour."""
    T = np.array(T, dtype=np.float32)
    for _ in range(layers):
        bad = np.isnan(T)
        if not bad.any():
            break
        P = np.pad(T, 1, constant_values=np.nan)
        s = np.zeros_like(T)
        n = np.zeros_like(T)
        for ax in range(T.ndim):
            for d in (-1, 1):
                sl = [slice(1, -1)] * T.ndim
                sl[ax] = slice(1 + d, P.shape[ax] - 1 + d)
                v = P[tuple(sl)]
                ok = ~np.isnan(v)
                s += np.where(ok, v, 0.0)
                n += ok
        T = np.where(bad & (n > 0), s / np.maximum(n, 1), T)
    return T


def _zlib_block(raw, level=6, block=1 << 20):
    """VTK's zlib appended-data layout: UInt64 header [n blocks, block size, last block size,
    compressed sizes...] followed by the compressed blocks."""
    import zlib
    parts = [zlib.compress(raw[i:i + block], level) for i in range(0, len(raw), block)] or [zlib.compress(b"")]
    last = len(raw) - block * (len(parts) - 1)
    head = np.array([len(parts), block, last if last != block else 0] + [len(c) for c in parts], dtype=np.uint64)
    return head.tobytes() + b"".join(parts)


def write_vti(path, point_data, origin, spacing, field_data=None, compress=False):
    """VTK XML ImageData with point data (one point per solver cell, at the cell centre: ``origin``
    is the centre of the first cell), raw appended binary (UInt64 headers). Arrays are (nx, ny, nz);
    an extent of 1 in z gives a 2-D image, which ParaView shows face-on as a coloured slice.
    ``compress``: zlib-compressed blocks (for the large 3-D snapshots)."""
    shape = next(iter(point_data.values())).shape
    nx, ny, nz = shape
    vtk_types = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.int8): "Int8",
                 np.dtype(np.uint8): "UInt8", np.dtype(np.int32): "Int32"}
    blocks, offset, arrays_xml = [], 0, []

    def add(name, a, ncomp=1, tag="DataArray"):
        nonlocal offset
        a = np.ascontiguousarray(a)
        raw = a.tobytes()
        b = _zlib_block(raw) if compress else np.uint64(len(raw)).tobytes() + raw
        blocks.append(b)
        arrays_xml.append(f'<{tag} type="{vtk_types[a.dtype]}" Name="{name}" NumberOfComponents="{ncomp}" '
                          f'NumberOfTuples="{a.size // ncomp}" format="appended" offset="{offset}"/>')
        offset += len(b)

    field_xml = ""
    if field_data:
        start = len(arrays_xml)
        for k, v in field_data.items():
            add(k, np.atleast_1d(np.asarray(v, dtype=np.float64)))
        field_xml = "<FieldData>" + "".join(arrays_xml[start:]) + "</FieldData>"
        del arrays_xml[start:]
    cell_start = len(arrays_xml)
    for k, v in point_data.items():
        # VTK image order: x fastest -> transpose (nx, ny, nz) to Fortran order
        add(k, np.asarray(v).transpose(2, 1, 0).ravel())
    cells_xml = "".join(arrays_xml[cell_start:])
    ext = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    head = (f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
            f'header_type="UInt64"{' compressor="vtkZLibDataCompressor"' if compress else ''}>\n<ImageData WholeExtent="{ext}" Origin="{origin[0]} {origin[1]} {origin[2]}" '
            f'Spacing="{spacing} {spacing} {spacing}">\n{field_xml}\n<Piece Extent="{ext}">\n'
            f'<PointData Scalars="{next(iter(point_data))}">{cells_xml}</PointData>\n</Piece>\n</ImageData>\n'
            f'<AppendedData encoding="raw">\n_')
    with open(path, "wb") as f:
        f.write(head.encode())
        for b in blocks:
            f.write(b)
        f.write(b"\n</AppendedData>\n</VTKFile>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="the run's JSON (tools/lbm/keyhole_runs/<tag>.json)")
    ap.add_argument("--out", default=None, help="output folder (default: next to the run)")
    ap.add_argument("--every", type=int, default=1, help="write every N-th saved section")
    ap.add_argument("--snap-every", type=int, default=1, help="export every N-th 3-D snapshot")
    ap.add_argument("--parts", default="sections,final,3d",
                    help="what to export, comma separated: sections, final, 3d (default: all there is)")
    ap.add_argument("--xz", action="store_true",
                    help="place the sections at their true position in 3-D (the x-z plane at the track's y), "
                         "to overlay them on the 3-D end state; default: flat 2-D images, height on the "
                         "vertical axis, which ParaView shows face-on")
    a = ap.parse_args()
    base = a.run[:-5] if a.run.endswith(".json") else a.run
    tag = os.path.basename(base)
    out = a.out or os.path.dirname(os.path.abspath(a.run))
    info = json.load(open(base + ".json"))
    dx = info["dx"] * 1e6
    z0 = -info["z_surface"] * 1e6
    log = info.get("log", [])
    by_t = {round(r["t"] * 1e9): r for r in log}
    written = []
    parts = set(a.parts.split(","))

    def fields(t):
        r = by_t.get(round(t * 1e9), {})
        # logs without ``laser_on`` (older runs): on while the laser position is logged
        on = r.get("laser_on", r.get("x_l") is not None) if r else False
        x_l = r.get("x_l") if r.get("x_l") is not None else r.get("x_last")   # x_last: after turn-off
        return dict(time_us=t * 1e6, laser_on=float(bool(on)), x_laser_um=x_l * 1e6 if x_l is not None else np.nan,
                    depth_um=r["depth"] * 1e6 if r.get("depth") is not None else np.nan)

    if "sections" in parts and os.path.exists(base + "_sections.npz"):
        with np.load(base + "_sections.npz") as zs:  # decompress each array once, not per frame
            s = {k: zs[k] for k in zs.files}
        d = os.path.join(out, f"{tag}_vtk")
        os.makedirs(d, exist_ok=True)
        planes = [("fill", "T", info["y"], "section", "")]
        if "fill_return" in s and "y_return" in info:      # turn runs: also the return track
            planes.append(("fill_return", "T_return", info["y_return"], "return", "_return"))
        for kf, kT, y_plane, stem, suffix in planes:
            j = int(round(y_plane / info["dx"]))
            entries = []
            for n in range(0, len(s["t"]), a.every):
                t = float(s["t"][n])
                fd = fields(t)
                fd["y_plane_um"] = (j + 0.5) * dx
                fn = f"{stem}_{n:04d}.vti"
                if a.xz:      # (nx, 1, nz) at y = the plane's cell centre
                    arr, org = (lambda v: v[:, None, :]), (0.5 * dx, (j + 0.5) * dx, z0 + 0.5 * dx)
                else:         # (nx, nz, 1): VTK x = x, VTK y = height z, a flat image
                    arr, org = (lambda v: v[:, :, None]), (0.5 * dx, z0 + 0.5 * dx, 0.0)
                write_vti(os.path.join(d, fn), dict(fill=arr(open_walls(s[kf][n])),
                                                    T=arr(s[kT][n]).astype(np.float32)), org, dx, fd)
                entries.append(f'<DataSet timestep="{t * 1e6:.4f}" part="0" file="{tag}_vtk/{fn}"/>')
            pvd = os.path.join(out, f"{tag}_section{suffix}.pvd")
            with open(pvd, "w") as f:
                f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
                        + "\n".join(entries) + "\n</Collection>\n</VTKFile>\n")
            written.append(f"{pvd} ({len(entries)} sections at y = {j * dx:.1f}..{(j + 1) * dx:.1f} um, time in us)")
    snap_dir = base + "_snap"
    if "3d" in parts and os.path.isdir(snap_dir):
        import glob
        files = sorted(glob.glob(os.path.join(snap_dir, "snap_*.npz")))[::max(1, a.snap_every)]
        d = os.path.join(out, f"{tag}_vtk")
        os.makedirs(d, exist_ok=True)
        entries = []
        extra = dict(y_track_um=info["y"] * 1e6)
        if "y_return" in info:
            extra["y_return_um"] = info["y_return"] * 1e6
        for n, fsnap in enumerate(files):
            with np.load(fsnap) as z:
                t = float(z["t"])
                fd = dict(fields(t), **extra)
                fn = f"snap_{os.path.basename(fsnap)[5:-4]}.vti"
                write_vti(os.path.join(d, fn), dict(fill=open_walls(z["fill"]), T=extend_into_gas(z["T"]),
                                                    flag=z["flag"].astype(np.int8)),
                          (0.5 * dx, 0.5 * dx, z0 + 0.5 * dx), dx, fd, compress=True)
            entries.append(f'<DataSet timestep="{t * 1e6:.4f}" part="0" file="{tag}_vtk/{fn}"/>')
            print(f"\r3-D snapshot {n + 1}/{len(files)}", end="", flush=True)
        if entries:
            print()
            pvd = os.path.join(out, f"{tag}_3d.pvd")
            with open(pvd, "w") as f:
                f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
                        + "\n".join(entries) + "\n</Collection>\n</VTKFile>\n")
            written.append(f"{pvd} ({len(entries)} 3-D snapshots, time in us)")
    if "final" in parts and os.path.exists(base + "_final.npz"):
        z = np.load(base + "_final.npz")
        fin = os.path.join(out, f"{tag}_final.vti")
        write_vti(fin, dict(fill=open_walls(z["fill"]), flag=z["flag"].astype(np.int8),
                            T_peak=z["T_peak"].astype(np.float32)), (0.5 * dx, 0.5 * dx, z0 + 0.5 * dx), dx,
                  dict(time_us=log[-1]["t"] * 1e6 if log else np.nan))
        written.append(fin)
    for w in written:
        print("wrote", w)
    if not written:
        sys.exit(f"nothing to export for {base}")


if __name__ == "__main__":
    main()
