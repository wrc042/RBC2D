import sys

sys.path.append("./")

from expio import Exporter

import taichi as ti
import numpy as np

from sim.interp import *
from sim.advect2d import *
from sim.fluid2d import *

from sim.projector2d import VelProject2D

from sim.init_fluid import init_leapfrog_vel_func


class Fluid2DSolver:
    def __init__(self, case=0):
        if case == 0:
            self._N = 256
            self._res = [self._N, self._N]
            res = self._res
            self._dx = 1.0 / self._N
            self._g = 9.81
            self._cfl = 1.0

        self._u_x = ti.field(float, [res[0] + 1, res[1]])
        self._u_y = ti.field(float, [res[0], res[1] + 1])
        self._u_x_tmp = ti.field(float, [res[0] + 1, res[1]])
        self._u_y_tmp = ti.field(float, [res[0], res[1] + 1])
        self._u_x_tmp2 = ti.field(float, [res[0] + 1, res[1]])
        self._u_y_tmp2 = ti.field(float, [res[0], res[1] + 1])

        self.projector = VelProject2D(res, self._dx, self._u_x, self._u_y)

        self._u_max = ti.field(float, shape=())
        self._u = ti.Vector.field(2, float, res)
        self._w = ti.field(float, res)
        self._t = 0.0

        if case == 0:
            self._vdt = 0.1
            self._total_frame = 50

            init_leapfrog_vel_func(self._u, self._dx)
            cal_u_split_kernel(self._u_x, self._u_y, self._u)
            self.projector.solve()

        self.export_data(0)

    def export_data(self, frame):
        cal_u_center_kernel(self._u_x, self._u_y, self._u)
        curl_2d_kernel(self._w, self._u, self._dx)
        w = self._w.to_numpy()
        Exporter.export_field_image(w, "vort", frame)
        Exporter.tag_frame(frame)

    def run(self):
        for i in range(1, self._total_frame + 1):
            target_t = self._vdt * i
            while self._t < target_t:
                max_velocity_2d_kernel(
                    self._u_x, self._u_y, self._u_max, self._res[0], self._res[1]
                )
                udt = self._cfl * self._dx / self._u_max[None]
                dt = udt
                if self._t + dt >= target_t:
                    dt = target_t - self._t
                elif (self._t + dt * 2) >= target_t:
                    dt = (target_t - self._t) * 0.5

                # substep
                self.substep(dt)

                self._t += dt
                Exporter.log(f"[FLUID] substep [{(self._vdt / dt):.2f}]")

            self.export_data(i)
        Exporter.log("[FLUID] done")

    def substep(self, dt):
        advect_u_mc_2d(
            self._u_x,
            self._u_y,
            self._u_x_tmp,
            self._u_y_tmp,
            self._u_x_tmp2,
            self._u_y_tmp2,
            self._dx,
            dt,
        )
        self.projector.solve()


if __name__ == "__main__":
    from ti_header import *

    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    Exporter.init("exps", "fluid")

    solver = Fluid2DSolver()
    solver.run()
