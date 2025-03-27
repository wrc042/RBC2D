import taichi as ti


@ti.kernel
def add_field_kernel(a: ti.template(), b: ti.template(), c: ti.template(), t: float):
    for idx in ti.grouped(a):
        c[idx] = a[idx] + t * b[idx]


@ti.kernel
def add_const_kernel(a: ti.template(), b: ti.template(), t: ti.template()):
    for idx in ti.grouped(a):
        b[idx] = a[idx] + t


@ti.kernel
def scale_field_kernel(a: ti.template(), b: ti.template(), t: float):
    for idx in ti.grouped(a):
        b[idx] = t * a[idx]
