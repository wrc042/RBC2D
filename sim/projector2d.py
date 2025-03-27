import taichi as ti

from sim.mgpcg import MGPCG


@ti.kernel
def enforce_boundary_kernel(u_x: ti.template(), u_y: ti.template()):
    for i, j in u_x:
        if i == 0 or i == u_x.shape[0] - 1:
            u_x[i, j] = 0.0

    for i, j in u_y:
        if j == 0 or j == u_y.shape[1] - 1:
            u_y[i, j] = 0.0


@ti.kernel
def set_rhs_kernel(
    u_x: ti.template(),
    u_y: ti.template(),
    u_div: ti.template(),
):
    for i, j in u_div:
        u_div[i, j] = u_x[i + 1, j] - u_x[i, j] + u_y[i, j + 1] - u_y[i, j]


@ti.kernel
def apply_pressure_kernel(
    u_x: ti.template(),
    u_y: ti.template(),
    p: ti.template(),
):
    for i, j in u_x:
        if i != 0 and i != u_x.shape[0] - 1:
            u_x[i, j] -= p[i, j] - p[i - 1, j]
    for i, j in u_y:
        if j != 0 and j != u_y.shape[1] - 1:
            u_y[i, j] -= p[i, j] - p[i, j - 1]


class VelProject2D:
    def __init__(self, res, dx, u_x, u_y, base_level=3):
        self._res = res
        self._dx = dx
        self._u_x = u_x
        self._u_y = u_y

        self._u_div = ti.field(float, res)

        boundary_tyoes = ti.Matrix([[2, 2], [2, 2]], ti.i32)
        self._solver = MGPCG(boundary_tyoes, res, base_level=base_level)

    def solve(
        self,
        max_iters=400,
        verbose=False,
        rel_tol=1e-12,
        abs_tol=1e-14,
        eps=1e-20,
    ):
        enforce_boundary_kernel(self._u_x, self._u_y)
        set_rhs_kernel(self._u_x, self._u_y, self._u_div)
        self._solver.init(self._u_div, -1.0)
        self._solver.solve(
            max_iters=max_iters,
            eps=eps,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            verbose=verbose,
        )
        apply_pressure_kernel(self._u_x, self._u_y, self._solver.x)
        enforce_boundary_kernel(self._u_x, self._u_y)
