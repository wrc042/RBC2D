import sys

sys.path.append("./")

from expio import Exporter

import taichi as ti
import numpy as np

from sim.interp import *
from sim.advect2d import *
from sim.flui2d import *


def chess_pattern(res, block_size):
    pattern = np.zeros(res, dtype=np.float32)
    uidx = np.arange(res[0])
    vidx = np.arange(res[1])
    ublack = (uidx // block_size) % 2 == 0
    vblack = (vidx // block_size) % 2 == 0
    pattern[ublack[:, None] & vblack] = 1.0
    pattern[~ublack[:, None] & ~vblack] = 1.0
    return pattern


class Fluid2DSolver:
    def __init__(self, case=0):
        if case == 0:
            self._N = 256
            self._res = [self._N, self._N]
            res = self._res
            self._dx = 1.0 / self._N
            self._g = 9.81
            self._cfl = 1.0

            self._vdt = 0.01
            self._total_frame = 20

        self._u_x = ti.field(float, [res[0] + 1, res[1]])
        self._u_y = ti.field(float, [res[0], res[1] + 1])
        self._u_x_tmp = ti.field(float, [res[0] + 1, res[1]])
        self._u_y_tmp = ti.field(float, [res[0], res[1] + 1])
        self._u_x_tmp2 = ti.field(float, [res[0] + 1, res[1]])
        self._u_y_tmp2 = ti.field(float, [res[0], res[1] + 1])

        self._u_max = ti.field(float, shape=())
        self._u = ti.Vector.field(2, float, res)
        self._u_mag = ti.field(float, res)
        self._w = ti.field(float, res)
        self._t = 0.0

        if case == 0:
            self._color = ti.field(float, res)
            self._color_tmp = ti.field(float, res)
            self._color_tmp2 = ti.field(float, res)
            self._color.from_numpy(chess_pattern(res, 8))

            x_grid = np.linspace(0.0, 1.0, self._N + 1)
            y_grid = np.linspace(self._dx * 0.5, 1.0 - self._dx * 0.5, self._N)
            x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing="ij")

            u_x_np = np.exp(-((x_grid - 0.5) ** 2 + (y_grid - 0.5) ** 2) / 0.05)
            self._u_x.from_numpy(u_x_np)
            self._u_y.fill(0.0)

        self.export_data(0)

    def export_data(self, frame):
        if self._color is not None:
            color = self._color.to_numpy()
            Exporter.export_field_image(
                color, "color", frame, relative=False, vmin=0.0, vmax=1.0, cmap="gray"
            )
        u_x = self._u_x.to_numpy()
        Exporter.export_field_image(u_x, "u_x", frame)
        u_y = self._u_y.to_numpy()
        Exporter.export_field_image(u_y, "u_y", frame)
        cal_u_mag_kernel(self._u_x, self._u_y, self._u_mag)
        u_mag = self._u_mag.to_numpy()
        Exporter.export_field_image(u_mag, "u_mag", frame)

    def run(self):
        for i in range(1, self._total_frame):
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
        advect_u_sl_2d(
            self._u_x,
            self._u_y,
            self._u_x_tmp,
            self._u_y_tmp,
            self._dx,
            dt,
        )
        advect_q_sl_2d(
            self._u_x,
            self._u_y,
            self._color,
            self._color_tmp,
            self._dx,
            dt,
        )
        # advect_u_mc_2d(
        #     self._u_x,
        #     self._u_y,
        #     self._u_x_tmp,
        #     self._u_y_tmp,
        #     self._u_x_tmp2,
        #     self._u_y_tmp2,
        #     self._dx,
        #     dt,
        # )
        # advect_q_mc_2d(
        #     self._u_x,
        #     self._u_y,
        #     self._color,
        #     self._color_tmp,
        #     self._color_tmp2,
        #     self._dx,
        #     dt,
        # )


if __name__ == "__main__":
    from ti_header import *

    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    Exporter.init("exps", "fluid")

    solver = Fluid2DSolver()
    solver.run()
