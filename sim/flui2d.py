import taichi as ti

from sim.interp import lookup_2d, is_valid


@ti.kernel
def max_velocity_2d_kernel(
    u_x: ti.template(), u_y: ti.template(), u_max: ti.template(), resx: int, resy: int
):
    u_max[None] = 1e-3
    for i, j in ti.ndrange(resx, resy):
        u = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        vel = ti.sqrt(u * u + v * v)
        ti.atomic_max(u_max[None], vel)


@ti.kernel
def cal_u_center_kernel(u_x: ti.template(), u_y: ti.template(), u: ti.template()):
    for i, j in u:
        ux = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        uy = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        u[i, j] = ti.Vector([ux, uy])

@ti.kernel
def cal_u_mag_kernel(u_x: ti.template(), u_y: ti.template(), u: ti.template()):
    for i, j in u:
        ux = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        uy = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        u[i, j] = ti.sqrt(ux * ux + uy * uy)


@ti.kernel
def cal_u_split_kernel(u_x: ti.template(), u_y: ti.template(), u: ti.template()):
    for i, j in u_x:
        if i == 0:
            u_x[i, j] = u[i, j][0]
        elif i == u_x.shape[0] - 1:
            u_x[i, j] = u[i - 1, j][0]
        else:
            u_x[i, j] = 0.5 * (u[i, j][0] + u[i - 1, j][0])

    for i, j in u_y:
        if j == 0:
            u_y[i, j] = u[i, j][1]
        elif j == u_y.shape[1] - 1:
            u_y[i, j] = u[i, j - 1][1]
        else:
            u_y[i, j] = 0.5 * (u[i, j][1] + u[i, j - 1][1])


@ti.kernel
def curl_2d_kernel(w: ti.template(), u: ti.template(), dx: float):
    for i, j in w:
        ul = lookup_2d(u, i - 1, j)
        ur = lookup_2d(u, i + 1, j)
        vb = lookup_2d(u, i, j - 1)
        vt = lookup_2d(u, i, j + 1)
        w[i, j] = (ur[1] - ul[1] - vt[0] + vb[0]) / (2 * dx)
