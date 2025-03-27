import taichi as ti

from sim.interp import *

from utils.ti_utils import *


# ODE
@ti.func
def advance(u_x, u_y, p, dx, dt):
    ret = p
    u0 = interp_umac_2_2d(u_x, u_y, p, dx)
    p1 = p + dt * u0 * 0.5
    u1 = interp_umac_2_2d(u_x, u_y, p1, dx)
    ret = p + dt * u1
    return ret
    # u0 = interp_umac_2_2d(u_x, u_y, p, dx)
    # p1 = p + dt * u0 * 0.5
    # u1 = interp_umac_2_2d(u_x, u_y, p1, dx)
    # p2 = p + dt * u1 * 0.5
    # u2 = interp_umac_2_2d(u_x, u_y, p2, dx)
    # p3 = p + dt * u2
    # u3 = interp_umac_2_2d(u_x, u_y, p3, dx)
    # ret = p + dt / 6.0 * (u0 + 2.0 * u1 + 2.0 * u2 + u3)
    # return ret

@ti.kernel
def advect_q_sl_2d_kernel(
    u_x: ti.template(),
    u_y: ti.template(),
    q: ti.template(),
    q_tmp: ti.template(),
    dx: float,
    dt: float,
    ofx: float,
    ofy: float,
):
    for i, j in q:
        p0 = ti.Vector([i + ofx, j + ofy]) * dx
        p = advance(u_x, u_y, p0, dx, -dt)
        q_tmp[i, j] = interp_2_2d(q, p, dx, ofx, ofy)

# semi-Lagrangian advection
def advect_q_sl_2d(u_x, u_y, q, q_tmp, dx, dt, ofx=0.5, ofy=0.5):
    advect_q_sl_2d_kernel(u_x, u_y, q, q_tmp, dx, dt, ofx, ofy)
    q.copy_from(q_tmp)

def advect_u_sl_2d(u_x, u_y, u_x_tmp, u_y_tmp, dx, dt):
    advect_q_sl_2d_kernel(u_x, u_y, u_x, u_x_tmp, dx, dt, 0.0, 0.5)
    advect_q_sl_2d_kernel(u_x, u_y, u_y, u_y_tmp, dx, dt, 0.5, 0.0)
    u_x.copy_from(u_x_tmp)
    u_y.copy_from(u_y_tmp)


# MacCormack scheme
def advect_q_mc_2d(u_x, u_y, q, q_tmp, q_tmp2, dx, dt, ofx=0.5, ofy=0.5):
    advect_q_sl_2d_kernel(u_x, u_y, q, q_tmp, dx, dt, ofx, ofy)
    advect_q_sl_2d_kernel(u_x, u_y, q_tmp, q_tmp2, dx, -dt, ofx, ofy)
    add_field_kernel(q, q_tmp2, q_tmp2, -1.0)
    add_field_kernel(q_tmp, q_tmp2, q, 0.5)


def advect_u_mc_2d(u_x, u_y, u_x_tmp, u_y_tmp, u_x_tmp2, u_y_tmp2, dx, dt):
    advect_q_sl_2d_kernel(u_x, u_y, u_x, u_x_tmp, dx, dt, 0.0, 0.5)
    advect_q_sl_2d_kernel(u_x, u_y, u_y, u_y_tmp, dx, dt, 0.5, 0.0)
    advect_q_sl_2d_kernel(u_x, u_y, u_x_tmp, u_x_tmp2, dx, -dt, 0.0, 0.5)
    advect_q_sl_2d_kernel(u_x, u_y, u_y_tmp, u_y_tmp2, dx, -dt, 0.5, 0.0)
    add_field_kernel(u_x, u_x_tmp2, u_x_tmp2, -1.0)
    add_field_kernel(u_y, u_y_tmp2, u_y_tmp2, -1.0)
    add_field_kernel(u_x_tmp, u_x_tmp2, u_x, 0.5)
    add_field_kernel(u_y_tmp, u_y_tmp2, u_y, 0.5)
