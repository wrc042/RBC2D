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

from sim.heat2d import *


class Fluid2DSolver:
    def __init__(self, args):
        case = args.c
        if case == 0:
            self._N = args.n
            self._res = [self._N * 3, self._N]
            res = self._res
            self._L = 0.05
            self._dx = self._L / self._N
            self._g = 9.81
            self._cfl = 1.0

        self._u_x = ti.field(float, [res[0] + 1, res[1]])
        self._u_y = ti.field(float, [res[0], res[1] + 1])
        self._u_x_tmp = ti.field(float, [res[0] + 1, res[1]])
        self._u_y_tmp = ti.field(float, [res[0], res[1] + 1])
        self._u_x_tmp2 = ti.field(float, [res[0] + 1, res[1]])
        self._u_y_tmp2 = ti.field(float, [res[0], res[1] + 1])

        self._T = ti.field(float, res)
        self._T_tmp = ti.field(float, res)
        self._T_tmp2 = ti.field(float, res)

        self.projector = VelProject2D(res, self._dx, self._u_x, self._u_y)

        self._u_max = ti.field(float, shape=())
        self._u = ti.Vector.field(2, float, res)
        self._u_mag = ti.field(float, res)
        self._w = ti.field(float, res)
        self._t = 0.0

        if case == 0:
            self._vdt = 0.2
            self._total_frame = args.f

            self._kappa = args.k
            self._alpha = args.a
            T0 = args.T0
            T1 = args.T1
            self._T0 = T0
            self._T1 = T1
            T = np.linspace(T0, T1, res[1], endpoint=True).reshape(1, -1)
            T = np.repeat(T, res[0], axis=0)
            width = self._N // 16
            T[
                T.shape[0] // 2 - width : T.shape[0] // 2 + width,
                T.shape[1] // 2 - width : T.shape[1] // 2 + width,
            ] = T0
            self._T.from_numpy(T)

            self._nu = args.nu
            self._u_x.fill(0.0)
            self._u_y.fill(0.0)

        Ra = (self._g * self._alpha * (self._T0 - self._T1) * self._L**3) / (
            self._nu * self._kappa
        )
        Exporter.log(f"[RBC] Ra = {Ra:.3e}")

        self.export_data(0)

    def export_data(self, frame):
        T = self._T.to_numpy()
        Exporter.export_field_image(T, "T", frame, dpi=200)

        Exporter.tag_frame(frame)

    def run(self):
        for i in range(1, self._total_frame + 1):
            target_t = self._vdt * i
            while self._t < target_t:
                max_velocity_2d_kernel(
                    self._u_x, self._u_y, self._u_max, self._res[0], self._res[1]
                )
                udt = self._cfl * self._dx / self._u_max[None]
                hdt = self._cfl * heat_diffuse_2d_cfl_dt(self._dx, self._kappa)
                nudt = self._cfl * viscosity_diffuse_2d_cfl_dt(self._dx, self._nu)
                dt = np.min([udt, hdt, nudt])
                Exporter.log(
                    f"[RBC] cfl vel [{(self._vdt / udt):.2f}] heat [{(self._vdt / hdt):.2f}] nu [{(self._vdt / nudt):.2f}]"
                )
                if self._t + dt >= target_t:
                    dt = target_t - self._t
                elif (self._t + dt * 2) >= target_t:
                    dt = (target_t - self._t) * 0.5

                # substep
                self.substep(dt)

                self._t += dt
                Exporter.log(f"[RBC] substep [{(self._vdt / dt):.2f}]")

            self.export_data(i)
        Exporter.log("[RBC] done")

    def substep(self, dt):
        advect_q_mc_2d(
            self._u_x,
            self._u_y,
            self._T,
            self._T_tmp,
            self._T_tmp2,
            self._dx,
            dt,
        )

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

        enforce_heat_bc_kernel(self._T, self._T0, self._T1)
        heat_diffuse_2d(
            self._T,
            self._T_tmp,
            self._kappa,
            self._dx,
            dt,
        )
        enforce_heat_bc_kernel(self._T, self._T0, self._T1)

        apply_gravity(
            self._u_y,
            self._T,
            self._alpha,
            self._g,
            self._T0,
            dt,
        )

        self.projector.solve()

        viscosity_diffuse_2d(
            self._u_x, self._u_y, self._u_x_tmp, self._u_y_tmp, self._nu, self._dx, dt
        )

        self.projector.solve()


if __name__ == "__main__":
    from ti_header import *

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=int, default=0, help="case")
    parser.add_argument("-n", type=int, default=128, help="resolution")
    parser.add_argument("-f", type=int, default=400, help="total frame")
    parser.add_argument("-k", type=float, default=5e-6, help="kappa")
    parser.add_argument("-a", type=float, default=2e-5, help="alpha")
    parser.add_argument("--nu", type=float, default=5e-6, help="nu")
    parser.add_argument("--T0", type=float, default=298, help="T0")
    parser.add_argument("--T1", type=float, default=273, help="T1")
    args = parser.parse_args()

    Exporter.init("exps", f"rbc-c{args.c}")

    solver = Fluid2DSolver(args)
    solver.run()
