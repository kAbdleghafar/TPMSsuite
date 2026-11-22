"""Export a keyhole run (tools/lbm/keyhole_runs/<tag>.json, _sections.npz, _final.npz) to VTK
files for ParaView. No extra dependencies: VTK XML ImageData with raw appended data.

Writes, next to the run (or to ``--out``):
  <tag>_vtk/section_0000.vti ...   the centreline x-z section every saved step (``fill`` = metal
                                    fill level, ``T`` [degC]; field data ``time_us``,
                                    ``laser_on``, ``x_laser_um``, ``depth_um``, ``y_plane_um``)
  <tag>_section.pvd                 the time series of those sections (open this one)
  <tag>_section_return.pvd          turn-point runs: the same on the return track
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


def write_vti(path, point_data, origin, spacing, field_data=None):
    """VTK XML ImageData with point data (one point per solver cell, at the cell centre: ``origin``
    is the centre of the first cell), raw appended binary (UInt64 headers). Arrays are (nx, ny, nz);
    an extent of 1 in z gives a 2-D image, which ParaView shows face-on as a coloured slice."""
    shape = next(iter(point_data.values())).shape
    nx, ny, nz = shape
    vtk_types = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.int8): "Int8",
                 np.dtype(np.uint8): "UInt8", np.dtype(np.int32): "Int32"}
    blocks, offset, arrays_xml = [], 0, []

    def add(name, a, ncomp=1, tag="DataArray"):
        nonlocal offset
        a = np.ascontiguousarray(a)
        raw = a.tobytes()
        blocks.append(np.uint64(len(raw)).tobytes() + raw)
        arrays_xml.append(f'<{tag} type="{vtk_types[a.dtype]}" Name="{name}" NumberOfComponents="{ncomp}" '
                          f'NumberOfTuples="{a.size // ncomp}" format="appended" offset="{offset}"/>')
        offset += 8 + len(raw)

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
            f'header_type="UInt64">\n<ImageData WholeExtent="{ext}" Origin="{origin[0]} {origin[1]} {origin[2]}" '
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
    if os.path.exists(base + "_sections.npz"):
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
                r = by_t.get(round(t * 1e9), {})
                # logs without ``laser_on`` (older runs): on while the laser position is logged
                on = r.get("laser_on", r.get("x_l") is not None) if r else False
                x_l = r.get("x_l") if r.get("x_l") is not None else r.get("x_last")   # x_last: after turn-off
                fd = dict(time_us=t * 1e6, laser_on=float(bool(on)),
                          x_laser_um=x_l * 1e6 if x_l is not None else np.nan,
                          depth_um=r["depth"] * 1e6 if r.get("depth") is not None else np.nan)
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
    if os.path.exists(base + "_final.npz"):
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
