import numpy as np
from cylbm.constants import cs, inv_cs2, inv_cs4
from cylbm.stencil import Stencil
import cupy as cp


class Lattice:
    def __init__(self, n, stencil: Stencil):
        self.stencil = stencil
        self.n = n
        self.f = cp.zeros((n + (stencil.q,)), dtype=cp.float64)
        self.f_tmp = cp.zeros_like(self.f)  # Pre-allocate buffer for streaming
        self.u = cp.zeros((n + (stencil.d,)), dtype=cp.float64)
        self.rho = cp.ones(n, dtype=cp.float64)
        self.gamma = 1.4
        
        # Pre-compute stencil arrays on GPU (cast to float for einsum)
        self.c_float = self.stencil.c.astype(cp.float64)
        self.w = self.stencil.w
        
        # Pre-compute streaming indices for faster streaming (avoid cp.roll)
        self._precompute_streaming_indices()

    def _precompute_streaming_indices(self):
        """Pre-compute indices for streaming to avoid cp.roll calls"""
        d = self.stencil.d
        n = self.n
        c = cp.asnumpy(self.stencil.c)  # Get numpy array for indexing computation
        
        if d == 2:
            nx, ny = n
            # Create base indices
            ix = np.arange(nx)
            iy = np.arange(ny)
            self.src_indices = []
            for iq in range(self.stencil.q):
                cx, cy = c[iq]
                src_x = (ix - cx) % nx
                src_y = (iy - cy) % ny
                self.src_indices.append((cp.asarray(src_x), cp.asarray(src_y)))
        elif d == 3:
            nx, ny, nz = n
            ix = np.arange(nx)
            iy = np.arange(ny)
            iz = np.arange(nz)
            self.src_indices = []
            for iq in range(self.stencil.q):
                cx, cy, cz = c[iq]
                src_x = (ix - cx) % nx
                src_y = (iy - cy) % ny
                src_z = (iz - cz) % nz
                self.src_indices.append((cp.asarray(src_x), cp.asarray(src_y), cp.asarray(src_z)))

    def init_data(self):
        self.f = self.feq()
        self.f_tmp = cp.zeros_like(self.f)

    def print_f(self):
        if self.stencil.d == 2:
            for j in range(self.f.shape[1]):
                for i in range(self.f.shape[0]):
                    ff = [f"{self.f[i, j, iq]}" for iq in range(self.stencil.q)]
                    print(f"[{i}, {j}] {ff}")
        elif self.stencil.d == 3:
            for k in range(self.f.shape[2]):
                for j in range(self.f.shape[1]):
                    for i in range(self.f.shape[0]):
                        ff = [f"{self.f[i, j, k, iq]}" for iq in range(self.stencil.q)]
                        print(f"[{i}, {j}, {k}] {ff}")

    def streaming(self):
        """Optimized streaming using pre-computed indices"""
        d = self.stencil.d
        
        if d == 2:
            for iq in range(self.stencil.q):
                src_x, src_y = self.src_indices[iq]
                self.f_tmp[:, :, iq] = self.f[src_x[:, None], src_y[None, :], iq]
        elif d == 3:
            for iq in range(self.stencil.q):
                src_x, src_y, src_z = self.src_indices[iq]
                self.f_tmp[:, :, :, iq] = self.f[src_x[:, None, None], src_y[None, :, None], src_z[None, None, :], iq]
        
        # Swap buffers
        self.f, self.f_tmp = self.f_tmp, self.f

    def density(self):
        """Optimized density calculation - already vectorized"""
        self.rho = cp.sum(self.f, axis=-1)

    def velocity(self):
        """Optimized velocity calculation using einsum"""
        # f shape: (nx, ny, [nz,] q), c_float shape: (q, d)
        # Result shape: (nx, ny, [nz,] d)
        if self.stencil.d == 2:
            # einsum: sum over q dimension, output (nx, ny, d)
            self.u = cp.einsum('ijq,qd->ijd', self.f, self.c_float) / self.rho[..., None]
        elif self.stencil.d == 3:
            # einsum: sum over q dimension, output (nx, ny, nz, d)
            self.u = cp.einsum('ijkq,qd->ijkd', self.f, self.c_float) / self.rho[..., None]

    def collision(self, omega):
        """Collision step with fused equilibrium calculation"""
        self.density()
        self.velocity()
        feq = self._feq_vectorized()
        self.f -= omega * (self.f - feq)

    def feq(self):
        """Public interface for equilibrium distribution"""
        return self._feq_vectorized()

    def _feq_vectorized(self):
        """Fully vectorized equilibrium distribution calculation"""
        # u shape: (nx, ny, [nz,] d), c_float shape: (q, d)
        # uc shape: (nx, ny, [nz,] q) - dot product of u with each velocity
        if self.stencil.d == 2:
            uc = cp.einsum('ijd,qd->ijq', self.u, self.c_float)
        elif self.stencil.d == 3:
            uc = cp.einsum('ijkd,qd->ijkq', self.u, self.c_float)
        
        # uu: velocity magnitude squared, shape (nx, ny, [nz,])
        uu = cp.sum(self.u**2, axis=-1)
        
        # Vectorized feq calculation
        # w shape: (q,), broadcast to match feq shape
        # rho shape: (nx, ny, [nz,]), broadcast with [None] for q dimension
        feq = self.w * self.rho[..., None] * (
            1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc**2 - 0.5 * inv_cs2 * uu[..., None]
        )
        return feq
