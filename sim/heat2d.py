import taichi as ti

from sim.interp import lookup_2d, is_valid


@ti.kernel
def heat_diffuse_2d_kernel(
    T: ti.template(), T_tmp: ti.template(), kappa: float, dx: float, dt: float
):
    fact = kappa * dt / dx / dx
    for i, j in T:
        T_tmp[i, j] = T[i, j] + fact * (
            lookup_2d(T, i - 1, j)
            + lookup_2d(T, i + 1, j)
            + lookup_2d(T, i, j - 1)
            + lookup_2d(T, i, j + 1)
            - 4 * T[i, j]
        )


def heat_diffuse_2d(T, T_tmp, kappa, dx, dt):
    heat_diffuse_2d_kernel(T, T_tmp, kappa, dx, dt)
    T.copy_from(T_tmp)


@ti.kernel
def enforce_heat_bc_kernel(
    T: ti.template(),
    T0: float,
    T1: float,
):
    for i, j in T:
        if j == 0:
            T[i, j] = T0
        elif j == T.shape[1] - 1:
            T[i, j] = T1


def heat_diffuse_2d_cfl_dt(dx, kappa, dim=2):
    return dx * dx / (2 * dim * kappa)


@ti.kernel
def apply_gravity(
    u_y: ti.template(),
    T: ti.template(),
    alpha: float,
    g: float,
    T0: float,
    dt: float,
):
    for i, j in T:
        u_y[i, j] -= g * dt * (1 - alpha * (T[i, j] - T0))
