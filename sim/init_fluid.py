import taichi as ti


# single vortex fields
@ti.func
def angular_vel_func(r, rad=0.02, strength=-0.01):
    r = r + 1e-6
    linear_vel = strength * 1.0 / r * (1.0 - ti.exp(-(r**2) / (rad**2)))
    return 1.0 / r * linear_vel


# vortex velocity field
@ti.kernel
def init_leapfrog_vel_func(vf: ti.template(), dx: float):
    c1 = ti.Vector([0.25, 0.62])
    c2 = ti.Vector([0.25, 0.38])
    c3 = ti.Vector([0.25, 0.74])
    c4 = ti.Vector([0.25, 0.26])
    w1 = -0.5
    w2 = 0.5
    w3 = -0.5
    w4 = 0.5
    for i, j in vf:
        # c1
        p = ti.Vector([i + 0.5, j + 0.5]) * dx - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition
        # c2
        p = ti.Vector([i + 0.5, j + 0.5]) * dx - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w2 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c3
        p = ti.Vector([i + 0.5, j + 0.5]) * dx - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w3 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c4
        p = ti.Vector([i + 0.5, j + 0.5]) * dx - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w4 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
