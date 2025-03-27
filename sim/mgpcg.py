import math
import time

import taichi as ti

# based on https://github.com/yitongdeng-projects/neural_flow_maps_code/blob/main/mgpcg.py


@ti.data_oriented
class MGPCG:
    """
    Grid-based MGPCG solver for the possion equation.

    .. note::

        This solver only runs on CPU and CUDA backends since it requires the
        ``pointer`` SNode.
    """

    def __init__(self, boundary_types, N, dim=2, base_level=3, real=float):
        """
        :parameter dim: Dimensionality of the fields.
        :parameter N: Grid resolutions.
        :parameter n_mg_levels: Number of multigrid levels.
        """

        # grid parameters
        self.use_multigrid = True

        self.N = N
        self.n_mg_levels = int(math.log2(min(N))) - base_level + 1
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 20
        self.dim = dim
        self.real = real

        self.r = [
            ti.field(dtype=real, shape=[n // (2**l) for n in self.N])
            for l in range(self.n_mg_levels)
        ]  # residual
        self.z = [
            ti.field(dtype=real, shape=[n // (2**l) for n in self.N])
            for l in range(self.n_mg_levels)
        ]  # M^-1 self.r
        self.x = ti.field(dtype=self.real, shape=self.N)  # solution
        self.p = ti.field(dtype=self.real, shape=self.N)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real, shape=self.N)  # matrix-vector product
        self.alpha = ti.field(dtype=self.real, shape=())  # step size
        self.beta = ti.field(dtype=self.real, shape=())  # step size
        self.sum = ti.field(dtype=self.real, shape=())  # storage for reductions
        self.r_mean = ti.field(dtype=self.real, shape=())  # storage for avg of r
        self.num_entries = math.prod(self.N)

        # boundaries: 1 for Dirichlet, 2 for Neumann
        self.boundary_types = boundary_types

    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        """
        Set up the solver for $\nabla^2 x = k r$, a scaled Poisson problem.
        :parameter k: (scalar) A scaling factor of the right-hand side.
        :parameter r: (ti.field) Unscaled right-hand side.
        """
        for I in ti.grouped(ti.ndrange(*self.N)):
            self.init_r(I, r[I] * k)

    @ti.kernel
    def get_result(self, x: ti.template()):
        """
        Get the solution field.

        :parameter x: (ti.field) The field to store the solution
        """
        for I in ti.grouped(ti.ndrange(*self.N)):
            x[I] = self.x[I]

    @ti.func
    def neighbor_sum(self, x, I):
        dims = x.shape
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # add right if has right
            if I[i] < dims[i] - 1:
                ret += x[I + offset]
            # add left if has left
            if I[i] > 0:
                ret += x[I - offset]
        return ret

    @ti.func
    def num_fluid_neighbors(self, x, I):
        dims = x.shape
        num = 2.0 * self.dim
        for i in ti.static(range(self.dim)):
            if I[i] <= 0 and (self.boundary_types[i, 0] == 2):
                num -= 1.0
            if I[i] >= dims[i] - 1 and (self.boundary_types[i, 1] == 2):
                num -= 1.0
        return num

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            multiplier = self.num_fluid_neighbors(self.p, I)
            self.Ap[I] = multiplier * self.p[I] - self.neighbor_sum(self.p, I)

    @ti.kernel
    def get_Ap(self, p: ti.template(), Ap: ti.template()):
        for I in ti.grouped(Ap):
            multiplier = self.num_fluid_neighbors(p, I)
            Ap[I] = multiplier * p[I] - self.neighbor_sum(p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            multiplier = self.num_fluid_neighbors(self.z[l], I)
            res = self.r[l][I] - (
                multiplier * self.z[l][I] - self.neighbor_sum(self.z[l], I)
            )
            self.r[l + 1][I // 2] += res * 1.0 / (self.dim - 1.0)

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                multiplier = self.num_fluid_neighbors(self.z[l], I)
                self.z[l][I] = (
                    self.r[l][I] + self.neighbor_sum(self.z[l], I)
                ) / multiplier

    @ti.kernel
    def recenter(self, r: ti.template()):  # so that the mean value of r is 0
        self.r_mean[None] = 0.0
        for I in ti.grouped(r):
            self.r_mean[None] += r[I] / self.num_entries
        for I in ti.grouped(r):
            r[I] -= self.r_mean[None]

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(
        self, max_iters=-1, eps=1e-12, abs_tol=1e-12, rel_tol=1e-12, verbose=False
    ):
        """
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        """
        all_neumann = self.boundary_types.sum() == 2 * 2 * self.dim

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p

        if all_neumann:
            self.recenter(self.r[0])
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        tol = max(abs_tol, old_zTr * rel_tol)

        # Conjugate gradients
        it = 0
        ti.sync()
        start_t = time.time()
        while max_iters == -1 or it < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f"iter {it}, |residual|_2={math.sqrt(rTr)}")

            if rTr < tol:
                ti.sync()
                end_t = time.time()
                print(
                    "[MGPCG] Converged at iter: ",
                    it,
                    " with final error: ",
                    math.sqrt(rTr),
                    " using time: ",
                    end_t - start_t,
                )
                return

            if all_neumann:
                self.recenter(self.r[0])
            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            it += 1

        ti.sync()
        end_t = time.time()
        print(
            "[MGPCG] Return without converging at iter: ",
            it,
            " with final error: ",
            math.sqrt(rTr),
            " using time: ",
            end_t - start_t,
        )
