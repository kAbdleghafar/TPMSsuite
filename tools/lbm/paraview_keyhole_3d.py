"""ParaView: movie of a keyhole run's 3-D snapshots, cut along the track: the metal (plate,
powder, melt) coloured by temperature, the keyhole and the pores as hollows in the cut face.

Needs a run made with ``--snapshots US`` and exported with keyhole_to_vtk.py (which writes
``<tag>_3d.pvd``). Three ways to use it:

  * pvpython, a movie (no window):
        pvpython tools/lbm/paraview_keyhole_3d.py path/to/<tag>_3d.pvd --movie keyhole.avi
    (.avi or .ogv; a .png name writes numbered frames). On a machine without a display:
        xvfb-run -a pvpython tools/lbm/paraview_keyhole_3d.py ... --movie keyhole.avi
    An MP4 (plays everywhere, small) from numbered frames:
        pvpython tools/lbm/paraview_keyhole_3d.py <tag>_3d.pvd --movie frames/f.png
        ffmpeg -framerate 12 -pattern_type glob -i "frames/f*.png" -vf "scale=1200:-2,format=yuv420p" -c:v libx264 keyhole.mp4
  * pvpython, interactive window:   pvpython tools/lbm/paraview_keyhole_3d.py path/to/<tag>_3d.pvd
  * ParaView GUI: View > Python Shell > Run Script, pick this file (set PVD below first).

Options:
  --view side      (default) looking at the cut face square on, the laser moving left to right,
                   no perspective
  --view oblique   from above and in front of the cut face, in perspective
  --return         cut along the return track of a turn-point run
  --title TEXT     first line of the annotation (default: the run's tag)
  --size W,H       image size in pixels (default 1600,600 side, 1200,900 oblique)
  --frame N        write only frame N as a .png (with --movie name.png): a quick check

Pipeline: the half of the domain from the track centreline outward (so the cut face is on the
track), the metal only (clip at 0.5 by ``surface``, the fill smoothed once, or ``fill`` in older
exports), its outer surface with smoothed normals, temperature in kelvin (``T`` + 273.15),
"Cool to Warm" colours from 300 to 4000 K, white background, and the text
"Grid: .. um   Time: .. microseconds / N closed gas pockets (hollows)".
"""
import glob
import os
import sys

from paraview.simple import (Calculator, Clip, ColorBy, ExtractSubset, ExtractSurface, GenerateSurfaceNormals,
                             GetActiveViewOrCreate, GetAnimationScene, GetColorTransferFunction, GetScalarBar,
                             PVDReader, PythonAnnotation, Render, SaveAnimation, SaveScreenshot, Show, servermanager)

PVD = None          # e.g. r"C:\runs\T3_dx2.5um_powder40um_f32_3d.pvd"
T_RANGE_K = (300.0, 4000.0)


def _try(obj, prop, *values):
    """Set a property that is spelt differently across ParaView versions."""
    for v in values:
        try:
            setattr(obj, prop, v)
            return True
        except (AttributeError, ValueError, TypeError):
            pass
    return False


def build(pvd, view_kind="side", plane="track", title=None, size=None):
    reader = PVDReader(FileName=pvd)
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    t0 = reader.TimestepValues[0] if reader.TimestepValues else 0.0
    reader.UpdatePipeline(t0)
    img = servermanager.Fetch(reader)
    x0, y0, z0 = img.GetOrigin()
    h = img.GetSpacing()[1]
    e = img.GetExtent()
    fd = img.GetFieldData()
    key = "y_return_um" if plane == "return" and fd.GetArray("y_return_um") else "y_track_um"
    y_cut = fd.GetArray(key).GetValue(0) if fd.GetArray(key) else y0 + 0.5 * (e[3] - e[2]) * h
    grid = fd.GetArray("grid_um").GetValue(0) if fd.GetArray("grid_um") else h
    j = max(e[2], min(e[3], int(round((y_cut - y0) / h))))

    # the half y >= y_cut: its cut face looks towards -y, so a camera on the -y side sees x
    # increasing to the right
    half = ExtractSubset(Input=reader)
    half.VOI = [e[0], e[1], j, e[3], e[4], e[5]]
    clip_by = "surface" if img.GetPointData().GetArray("surface") else "fill"   # smoothed fill if exported
    metal = Clip(Input=half)
    metal.ClipType = "Scalar"
    metal.Scalars = ["POINTS", clip_by]
    metal.Value = 0.5
    _try(metal, "Invert", 0)
    # which side a scalar clip keeps has changed between ParaView versions: keep the metal side
    metal.UpdatePipeline(t0)
    out = servermanager.Fetch(metal)
    arr = out.GetPointData().GetArray(clip_by) if out is not None else None
    if arr is not None and arr.GetNumberOfTuples() > 0:
        idx = range(0, arr.GetNumberOfTuples(), max(1, arr.GetNumberOfTuples() // 997))
        if sum(arr.GetValue(i) for i in idx) / len(idx) < 0.5:
            _try(metal, "Invert", 1)
    # a surface with consistent, smoothed normals: no dark facets on the clipped walls
    skin = GenerateSurfaceNormals(Input=ExtractSurface(Input=metal))
    _try(skin, "Splitting", 0)
    kelvin = Calculator(Input=skin)
    _try(kelvin, "AttributeType", "Point Data")
    kelvin.ResultArrayName = "Temperature (K)"
    kelvin.Function = "T+273.15"

    view = GetActiveViewOrCreate("RenderView")
    if size is None:
        size = [1600, 600] if view_kind == "side" else [1200, 900]
    view.ViewSize = list(size)
    view.OrientationAxesVisibility = 0
    _try(view, "UseColorPaletteForBackground", 0)
    view.Background = [1.0, 1.0, 1.0]
    _try(view, "BackgroundColorMode", "Single Color")
    disp = Show(kelvin, view)
    ColorBy(disp, ("POINTS", "Temperature (K)"))
    lut = GetColorTransferFunction("Temperature (K)")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(*T_RANGE_K)
    _try(lut, "AutomaticRescaleRangeMode", "Never")
    bar = GetScalarBar(lut, view)
    bar.Title = "Temperature (K)"
    bar.ComponentTitle = ""
    _try(bar, "Orientation", "Horizontal")
    _try(bar, "WindowLocation", "Lower Center", "LowerCenter")
    _try(bar, "ScalarBarLength", 0.5)
    for p in ("TitleFontSize", "LabelFontSize"):
        _try(bar, p, 18)
    for p in ("TitleColor", "LabelColor"):
        _try(bar, p, [0.0, 0.0, 0.0])
    _try(bar, "RangeLabelFormat", "%-#6.1e")
    disp.SetScalarBarVisibility(view, True)

    tag = os.path.basename(pvd).replace("_3d.pvd", "")
    head = title or f"TPMSsuite LBM {tag}"
    label = PythonAnnotation(Input=reader)
    _try(label, "ArrayAssociation", "Field Data")
    label.Expression = ("%r + '\\nGrid: %g um    Time: %%.0f microseconds\\n%%d closed gas pockets (hollows)' "
                        "%% (t_value, int(closed_pockets[0]) if 'closed_pockets' in dir() else 0)") % (head, grid)
    lab = Show(label, view)
    lab.FontSize = 20
    _try(lab, "Color", [0.0, 0.0, 0.0])
    _try(lab, "Justification", "Left")
    _try(lab, "WindowLocation", "Upper Left Corner", "UpperLeftCorner")

    cx = x0 + 0.5 * (e[1] - e[0]) * h
    cz = z0 + 0.5 * (e[5] - e[4]) * h
    L = (e[1] - e[0]) * h
    H = (e[5] - e[4]) * h
    view.CameraFocalPoint = [cx, y_cut, cz]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    if view_kind == "side":
        view.CameraPosition = [cx, y_cut - 3.0 * L, cz]
        view.CameraParallelProjection = 1
        view.ResetCamera(False)
        view.CameraParallelScale = 0.5 * max(H, L * size[1] / size[0]) * 1.08
    else:
        view.CameraPosition = [cx - 0.35 * L, y_cut - 1.6 * L, cz + 0.45 * L]
        view.CameraParallelProjection = 0
        view.ResetCamera(False)
    Render(view)
    return view, reader


def main(argv):
    def opt(name, default=None):
        if name in argv:
            i = argv.index(name)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    movie = opt("--movie")
    view_kind = opt("--view", "side")
    title = opt("--title")
    size = opt("--size")
    frame = opt("--frame")
    plane = "return" if "--return" in argv else "track"
    args = [a for a in argv if not a.startswith("--")]
    pvd = args[0] if args else PVD
    if not pvd:
        found = sorted(glob.glob("*_3d.pvd"), key=os.path.getmtime)
        if not found:
            raise SystemExit("give the <tag>_3d.pvd (or set PVD at the top of this script)")
        pvd = found[-1]
    view, reader = build(pvd, view_kind, plane, title, [int(v) for v in size.split(",")] if size else None)
    if movie and frame is not None:
        t = reader.TimestepValues[int(frame)]
        view.ViewTime = t
        GetAnimationScene().AnimationTime = t
        Render(view)
        SaveScreenshot(movie, view, ImageResolution=list(view.ViewSize))
        print("wrote", movie)
    elif movie:
        SaveAnimation(movie, view, ImageResolution=list(view.ViewSize), FrameRate=15)
        print("wrote", movie)
    elif len(sys.argv) > 1 and not hasattr(sys, "ps1"):
        from paraview.simple import Interact
        Interact(view)
    return view


if __name__ == "__main__":
    main(sys.argv[1:])
