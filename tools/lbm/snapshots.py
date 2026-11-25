"""3-D snapshots of a free-surface keyhole run, for movies in ParaView.

Every ``interval`` seconds of simulated time the whole domain is written to
``<out>/<tag>_snap/snap_<time in ns>.npz``: ``fill`` (metal fill level) and ``T`` [degC, NaN where
there is no metal], both float16, and ``flag`` (int8), compressed. One snapshot of a 5.6 M-cell
domain takes about 8-15 MB on disk, so 10 us snapshots of a 2.3 ms turn run take 2-3 GB.

The files are written while the run goes (nothing is held in memory) and are named by their time,
so a resumed run continues where it stopped and a snapshot written twice is simply replaced.
``keyhole_to_vtk.py`` turns them into a ParaView time series (``<tag>_3d.pvd``), and
``paraview_keyhole_3d.py`` sets up the cut view.
"""
import os

import numpy as np

from tpmsfem.backend import asnumpy


class Snapshots:
    def __init__(self, folder, interval, t_now=0.0):
        """``interval`` in seconds (0 or None: off). ``t_now``: the time the run starts or resumes
        at; the next snapshot is the first multiple of ``interval`` at or after it."""
        self.folder = folder
        self.interval = float(interval or 0.0)
        self.count = 0
        if self.interval > 0:
            os.makedirs(folder, exist_ok=True)
            self.next = np.ceil(t_now / self.interval - 1e-9) * self.interval
        else:
            self.next = np.inf

    def __call__(self, m, t):
        """Call from the run's callback; writes a snapshot when ``t`` has reached the next one."""
        if t + 1e-12 < self.next:
            return False
        metal = m.metal()
        fill = asnumpy(m.lbm.fill).astype(np.float16)
        T = np.where(asnumpy(metal), asnumpy(m.T), np.nan).astype(np.float16)
        np.savez_compressed(os.path.join(self.folder, f"snap_{int(round(t * 1e9)):09d}.npz"),
                            t=np.float64(t), fill=fill, T=T, flag=asnumpy(m.lbm.flag).astype(np.int8))
        self.count += 1
        while self.next <= t + 1e-12:
            self.next += self.interval
        return True
