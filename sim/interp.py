import taichi as ti

from utils.ti_utils import *


@ti.func
def N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 3.0 / 4.0 - abs_x**2
    elif abs_x < 1.5:
        result = 0.5 * (3.0 / 2.0 - abs_x) ** 2
    return result


@ti.func
def lookup_2d(f: ti.template(), u: int, v: int):
    size = f.shape
    i = ti.math.clamp(int(u), 0, size[0] - 1)
    j = ti.math.clamp(int(v), 0, size[1] - 1)
    return f[i, j]


@ti.func
def interp_2_2d(f, p, dx, ofx=0.5, ofy=0.5):
    eps = 1e-5
    size = f.shape

    u, v = p / dx
    u = u - ofx
    v = v - ofy
    u = ti.math.clamp(u, 1, size[0] - 2 - eps)
    v = ti.math.clamp(v, 1, size[1] - 2 - eps)

    gu = ti.floor(u)
    gv = ti.floor(v)
    iu = int(gu)
    iv = int(gv)

    val = 0.0

    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = u - (gu + i)
            y_p_y_j = v - (gv + j)
            val += N_2(x_p_x_i) * N_2(y_p_y_j) * lookup_2d(f, iu + i, iv + j)

    return val


@ti.func
def interp_umac_2_2d(u_x, u_y, p, dx):
    ux = interp_2_2d(u_x, p, dx, 0.0, 0.5)
    uy = interp_2_2d(u_y, p, dx, 0.5, 0.0)
    return ti.Vector([ux, uy])


@ti.func
def is_valid(res, idx):
    return all(idx >= 0) and all(idx < res)
