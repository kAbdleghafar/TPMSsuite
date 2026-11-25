"""ParaView: the cut view of a keyhole run's 3-D snapshots (a movie like a half-domain cut along
the track, metal coloured by temperature).

Needs a run made with ``--snapshots US`` and exported with keyhole_to_vtk.py (which writes
``<tag>_3d.pvd``). Three ways to use it:

  * ParaView GUI: View > Python Shell > Run Script, pick this file. It asks nothing: set PVD
    below to your <tag>_3d.pvd first (or leave it None and it takes the newest *_3d.pvd in the
    current folder, which is the folder ParaView was started from).
  * pvpython, interactive window:   pvpython tools/lbm/paraview_keyhole_3d.py path/to/<tag>_3d.pvd
  * pvpython, a movie (no window):  pvpython tools/lbm/paraview_keyhole_3d.py path/to/<tag>_3d.pvd --movie keyhole.avi
    (.avi, .ogv or a .png name for numbered frames; --return for the return track of a turn run)

Pipeline: the domain from y = 0 to the track centreline (so the cut face is on the track), then
the metal only (clip by ``fill`` at 0.5), coloured by ``T`` [degC], seen from the cut side.
"""
import glob
import os
import sys

from paraview.simple import (Clip, ColorBy, ExtractSubset, GetActiveViewOrCreate, GetAnimationScene,
                             GetColorTransferFunction, GetScalarBar, PVDReader, PythonAnnotation, Render,
                             SaveAnimation, Show, servermanager)

PVD = None          # e.g. r"C:\runs\M200_turn_dx3.33um_f32_3d.pvd"
T_RANGE = (25.0, 3500.0)
SIZE = [1600, 900]


def main(pvd, movie=None, plane="track"):
    reader = PVDReader(FileName=pvd)
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    reader.UpdatePipeline(reader.TimestepValues[0] if reader.TimestepValues else 0.0)
    img = servermanager.Fetch(reader)
    x0, y0, z0 = img.GetOrigin()
    h = img.GetSpacing()[1]
    e = img.GetExtent()
    fd = img.GetFieldData()
    key = "y_return_um" if plane == "return" and fd.GetArray("y_return_um") else "y_track_um"
    y_cut = fd.GetArray(key).GetValue(0) if fd.GetArray(key) else y0 + 0.5 * (e[3] - e[2]) * h
    j = int(round((y_cut - y0) / h))

    half = ExtractSubset(Input=reader)
    half.VOI = [e[0], e[1], e[2], j, e[4], e[5]]
    metal = Clip(Input=half)
    metal.ClipType = "Scalar"
    metal.Scalars = ["POINTS", "fill"]
    metal.Value = 0.5
    metal.Invert = 0
    # which side a scalar clip keeps has changed between ParaView versions: keep the metal side
    metal.UpdatePipeline(reader.TimestepValues[0] if reader.TimestepValues else 0.0)
    out = servermanager.Fetch(metal)
    arr = out.GetPointData().GetArray("fill") if out is not None else None
    if arr is not None and arr.GetNumberOfTuples() > 0:
        mean = sum(arr.GetValue(i) for i in range(0, arr.GetNumberOfTuples(), 97)) / len(range(0, arr.GetNumberOfTuples(), 97))
        if mean < 0.5:
            metal.Invert = 1

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = SIZE
    view.OrientationAxesVisibility = 1
    disp = Show(metal, view)
    ColorBy(disp, ("POINTS", "T"))
    lut = GetColorTransferFunction("T")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(*T_RANGE)
    try:
        lut.AutomaticRescaleRangeMode = "Never"            # ParaView 5.5+
    except AttributeError:
        pass
    bar = GetScalarBar(lut, view)
    bar.Title = "T [degC]"
    bar.ComponentTitle = ""
    disp.SetScalarBarVisibility(view, True)

    label = PythonAnnotation(Input=reader)
    label.Expression = "'t = %.1f us' % t_value"
    lab = Show(label, view)
    lab.FontSize = 14
    for loc in ("Upper Left Corner", "UpperLeftCorner"):  # the name changed in ParaView 5.9
        try:
            lab.WindowLocation = loc
            break
        except (ValueError, AttributeError):
            pass

    # look at the cut face from the +y side, a little from above and from the start of the track
    cx = x0 + 0.5 * (e[1] - e[0]) * h
    cz = z0 + 0.5 * (e[5] - e[4]) * h
    L = (e[1] - e[0]) * h
    view.CameraFocalPoint = [cx, y_cut, cz]
    view.CameraPosition = [cx - 0.35 * L, y_cut + 1.6 * L, cz + 0.45 * L]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.ResetCamera(False)
    Render(view)
    if movie:
        SaveAnimation(movie, view, ImageResolution=SIZE, FrameRate=15)
        print("wrote", movie)
    return view


def _find(pvd):
    if pvd:
        return pvd
    found = sorted(glob.glob("*_3d.pvd"), key=os.path.getmtime)
    if not found:
        raise SystemExit("set PVD at the top of this script to your <tag>_3d.pvd")
    return found[-1]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    movie = sys.argv[sys.argv.index("--movie") + 1] if "--movie" in sys.argv else None
    if movie in args:
        args.remove(movie)
    view = main(_find(args[0] if args else PVD), movie, "return" if "--return" in sys.argv else "track")
    if not movie and "paraview.simple" in sys.modules and not hasattr(sys, "ps1") and len(sys.argv) > 1:
        from paraview.simple import Interact
        Interact(view)
