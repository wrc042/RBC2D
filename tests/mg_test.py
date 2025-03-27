import sys

sys.path.append("./")

from ti_header import *

from expio import Exporter

from sim.mgpcg import MGPCG


# Dirichlet BC
# Poisson sin(kx x) * sin(ky y)
# [0, 1] x [0, 1]
@ti.kernel
def test2d_c0_init_kernel(
    rhs: ti.template(),
    dx: float,
    kx: float,
    ky: float,
):
    fact = dx * dx * (kx * kx + ky * ky)
    for i, j in rhs:
        x = (i + 1) * dx
        y = (j + 1) * dx
        rhs[i, j] = fact * ti.sin(kx * x) * ti.sin(ky * y)


# Neumann BC
# Poisson cos(kx x) * cos(ky y)
# [0, 1] x [0, 1]
@ti.kernel
def test2d_c1_init_kernel(
    rhs: ti.template(),
    dx: float,
    kx: float,
    ky: float,
):
    fact = dx * dx * (kx * kx + ky * ky)
    for i, j in rhs:
        x = (i + 0.5) * dx
        y = (j + 0.5) * dx
        rhs[i, j] = fact * ti.cos(kx * x) * ti.cos(ky * y)


# Hybrid BC
# Laplace (2x - 1)
# [0, 1] x [0, 1]
@ti.kernel
def test2d_c2_init_kernel(
    rhs: ti.template(),
):
    for i, j in rhs:
        rhs[i, j] = 0.0
        if i == 0:
            rhs[i, j] = -1.0
        elif i == rhs.shape[0] - 1:
            rhs[i, j] = 1.0


# Dirichlet BC
# Poisson sin(kx x) * sin(ky y) * sin(kz z)
# [0, 1]  x [0, 1] x [0, 1]
@ti.kernel
def test3d_c0_init_kernel(
    rhs: ti.template(),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
):
    fact = dx * dx * (kx * kx + ky * ky + kz * kz)
    for i, j, k in rhs:
        x = (i + 1) * dx
        y = (j + 1) * dx
        z = (k + 1) * dx
        rhs[i, j, k] = fact * ti.sin(kx * x) * ti.sin(ky * y) * ti.sin(kz * z)


# Neumann BC
# Poisson cos(kx x) * cos(ky y) * cos(kz z)
# [0, 1] x [0, 1] x [0, 1]
@ti.kernel
def test3d_c1_init_kernel(
    b: ti.template(),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
):
    fact = dx * dx * (kx * kx + ky * ky + kz * kz)
    for i, j, k in rhs:
        x = (i + 0.5) * dx
        y = (j + 0.5) * dx
        z = (k + 0.5) * dx
        rhs[i, j, k] = fact * ti.cos(kx * x) * ti.cos(ky * y) * ti.cos(kz * z)


# Hybrid BC
# Laplace (2x - 1)
# [0, 1] x [0, 1] x [0, 1]
@ti.kernel
def test3d_c2_init_kernel(
    rhs: ti.template(),
):
    for i, j, k in rhs:
        rhs[i, j, k] = 0.0
        if i == 0:
            rhs[i, j, k] = -1.0
        elif i == rhs.shape[0] - 1:
            rhs[i, j, k] = 1.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument("-c", type=int, default=0)
    parser.add_argument("--res", type=int, default=512)
    args = parser.parse_args()

    test_case = args.c
    dim = args.d
    res = args.res

    Exporter.init("exps", f"mgtest{dim}d-c{test_case}")

    if dim == 2:
        if test_case == 0:
            res = [res] * dim
            dx = 1.0 / (res[0] + 1)
            kx = 6 * ti.math.pi
            ky = 6 * ti.math.pi
            boundary_types = ti.Matrix([[1, 1], [1, 1]], ti.i32)

            x_grid = np.linspace(dx, 1 - dx, res[0])
            y_grid = np.linspace(dx, 1 - dx, res[1])
            x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing="ij")
            x_gt = np.sin(kx * x_grid) * np.sin(ky * y_grid)

            solver = MGPCG(boundary_types, res)
            rhs = ti.field(ti.f32, res)
            test2d_c0_init_kernel(rhs, dx, kx, ky)
            solver.init(rhs, 1.0)

        elif test_case == 1:
            res = [res] * dim
            dx = 1.0 / res[0]
            kx = 6 * ti.math.pi
            ky = 6 * ti.math.pi
            boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32)

            x_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[0])
            y_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[1])
            x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing="ij")
            x_gt = np.cos(kx * x_grid) * np.cos(ky * y_grid)

            solver = MGPCG(boundary_types, res)
            rhs = ti.field(ti.f32, res)
            test2d_c1_init_kernel(rhs, dx, kx, ky)
            solver.init(rhs, 1.0)

        elif test_case == 2:
            res = [res] * dim
            dx = 1.0 / res[0]
            boundary_types = ti.Matrix([[1, 1], [2, 2]], ti.i32)

            x_grid = np.linspace(dx, 1 - dx, res[0])
            y_grid = np.linspace(dx, 1 - dx, res[1])
            x_grid, y_grid = np.meshgrid(x_grid, y_grid, indexing="ij")
            x_gt = 2 * x_grid - 1

            solver = MGPCG(boundary_types, res)
            rhs = ti.field(ti.f32, res)
            test2d_c2_init_kernel(rhs)
            solver.init(rhs, 1.0)
    elif dim == 3:
        if test_case == 0:
            res = [res] * dim
            dx = 1.0 / (res[0] + 1)
            kx = 6 * ti.math.pi
            ky = 6 * ti.math.pi
            kz = 6 * ti.math.pi
            boundary_types = ti.Matrix([[1, 1], [1, 1], [1, 1]], ti.i32)

            x_grid = np.linspace(dx, 1 - dx, res[0])
            y_grid = np.linspace(dx, 1 - dx, res[1])
            z_grid = np.linspace(dx, 1 - dx, res[2])
            x_grid, y_grid, z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
            x_gt = np.sin(kx * x_grid) * np.sin(ky * y_grid) * np.sin(kz * z_grid)

            solver = MGPCG(boundary_types, res, 3)
            rhs = ti.field(ti.f32, res)
            test3d_c0_init_kernel(rhs, dx, kx, ky, kz)
            solver.init(rhs, 1.0)

        elif test_case == 1:
            res = [res] * dim
            dx = 1.0 / res[0]
            kx = 6 * ti.math.pi
            ky = 6 * ti.math.pi
            kz = 6 * ti.math.pi
            boundary_types = ti.Matrix([[2, 2], [2, 2], [2, 2]], ti.i32)

            x_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[0])
            y_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[1])
            z_grid = np.linspace(dx * 0.5, 1 - dx * 0.5, res[2])
            x_grid, y_grid, z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
            x_gt = np.cos(kx * x_grid) * np.cos(ky * y_grid) * np.cos(kz * z_grid)

            solver = MGPCG(boundary_types, res, 3)
            rhs = ti.field(ti.f32, res)
            test3d_c1_init_kernel(rhs, dx, kx, ky, kz)
            solver.init(rhs, 1.0)

        elif test_case == 2:
            res = [res] * dim
            dx = 1.0 / res[0]
            boundary_types = ti.Matrix([[1, 1], [2, 2], [2, 2]], ti.i32)

            x_grid = np.linspace(dx, 1 - dx, res[0])
            y_grid = np.linspace(dx, 1 - dx, res[1])
            z_grid = np.linspace(dx, 1 - dx, res[2])
            x_grid, y_grid, z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
            x_gt = 2 * x_grid - 1

            solver = MGPCG(boundary_types, res, 3)
            rhs = ti.field(ti.f32, res)
            test3d_c2_init_kernel(rhs)
            solver.init(rhs, 1.0)

    solver.solve(verbose=True)

    x = solver.x.to_numpy()

    if dim == 2:
        Exporter.export_field_image(x, "x")

    if x_gt is not None:
        if dim == 2:
            Exporter.export_field_image(x_gt, "x_gt")
        Exporter.log(f"L1 error: {np.mean(np.abs(x - x_gt)):.4e}")

    Exporter.tag_frame()
