import numpy as np
from numba import cuda
import math
from Nulbm.constants import cs, inv_cs2, inv_cs4
from Nulbm.stencil import Stencil


# ============== CUDA Kernels for D2Q9 ==============

@cuda.jit
def streaming_kernel_2d(f, f_new, c, nx, ny, q):
    """CUDA kernel for streaming in 2D with periodic boundary conditions."""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        for iq in range(q):
            # Source indices with periodic boundary
            src_i = (i - c[iq, 0] + nx) % nx
            src_j = (j - c[iq, 1] + ny) % ny
            f_new[i, j, iq] = f[src_i, src_j, iq]


@cuda.jit
def density_kernel_2d(f, rho, nx, ny, q):
    """CUDA kernel for computing density in 2D."""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        local_rho = 0.0
        for iq in range(q):
            local_rho += f[i, j, iq]
        rho[i, j] = local_rho


@cuda.jit
def velocity_kernel_2d(f, u, rho, c, nx, ny, q, d):
    """CUDA kernel for computing velocity in 2D."""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        inv_rho = 1.0 / rho[i, j]
        for dim in range(d):
            vel = 0.0
            for iq in range(q):
                vel += f[i, j, iq] * c[iq, dim]
            u[i, j, dim] = vel * inv_rho


@cuda.jit
def collision_kernel_2d(f, rho, u, c, w, omega, inv_cs2_val, inv_cs4_val, nx, ny, q, d):
    """CUDA kernel for BGK collision in 2D - computes feq and applies collision in one step."""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Compute density
        local_rho = 0.0
        for iq in range(q):
            local_rho += f[i, j, iq]
        rho[i, j] = local_rho
        
        # Compute velocity
        inv_rho = 1.0 / local_rho
        ux = 0.0
        uy = 0.0
        for iq in range(q):
            ux += f[i, j, iq] * c[iq, 0]
            uy += f[i, j, iq] * c[iq, 1]
        ux *= inv_rho
        uy *= inv_rho
        u[i, j, 0] = ux
        u[i, j, 1] = uy
        
        # Compute u*u for feq
        uu = ux * ux + uy * uy
        
        # Collision with feq computation
        for iq in range(q):
            uc = ux * c[iq, 0] + uy * c[iq, 1]
            feq = w[iq] * local_rho * (1.0 + inv_cs2_val * uc + 0.5 * inv_cs4_val * uc * uc - 0.5 * inv_cs2_val * uu)
            f[i, j, iq] = f[i, j, iq] - omega * (f[i, j, iq] - feq)


@cuda.jit
def feq_kernel_2d(f, feq, rho, u, c, w, inv_cs2_val, inv_cs4_val, nx, ny, q):
    """CUDA kernel for computing equilibrium distribution in 2D."""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        ux = u[i, j, 0]
        uy = u[i, j, 1]
        uu = ux * ux + uy * uy
        local_rho = rho[i, j]
        
        for iq in range(q):
            uc = ux * c[iq, 0] + uy * c[iq, 1]
            feq[i, j, iq] = w[iq] * local_rho * (1.0 + inv_cs2_val * uc + 0.5 * inv_cs4_val * uc * uc - 0.5 * inv_cs2_val * uu)


# ============== CUDA Kernels for D3Q19 ==============

@cuda.jit
def streaming_kernel_3d(f, f_new, c, nx, ny, nz, q):
    """CUDA kernel for streaming in 3D with periodic boundary conditions."""
    i, j, k = cuda.grid(3)
    if i < nx and j < ny and k < nz:
        for iq in range(q):
            src_i = (i - c[iq, 0] + nx) % nx
            src_j = (j - c[iq, 1] + ny) % ny
            src_k = (k - c[iq, 2] + nz) % nz
            f_new[i, j, k, iq] = f[src_i, src_j, src_k, iq]


@cuda.jit
def density_kernel_3d(f, rho, nx, ny, nz, q):
    """CUDA kernel for computing density in 3D."""
    i, j, k = cuda.grid(3)
    if i < nx and j < ny and k < nz:
        local_rho = 0.0
        for iq in range(q):
            local_rho += f[i, j, k, iq]
        rho[i, j, k] = local_rho


@cuda.jit
def velocity_kernel_3d(f, u, rho, c, nx, ny, nz, q, d):
    """CUDA kernel for computing velocity in 3D."""
    i, j, k = cuda.grid(3)
    if i < nx and j < ny and k < nz:
        inv_rho = 1.0 / rho[i, j, k]
        for dim in range(d):
            vel = 0.0
            for iq in range(q):
                vel += f[i, j, k, iq] * c[iq, dim]
            u[i, j, k, dim] = vel * inv_rho


@cuda.jit
def collision_kernel_3d(f, rho, u, c, w, omega, inv_cs2_val, inv_cs4_val, nx, ny, nz, q, d):
    """CUDA kernel for BGK collision in 3D."""
    i, j, k = cuda.grid(3)
    if i < nx and j < ny and k < nz:
        # Compute density
        local_rho = 0.0
        for iq in range(q):
            local_rho += f[i, j, k, iq]
        rho[i, j, k] = local_rho
        
        # Compute velocity
        inv_rho = 1.0 / local_rho
        ux = 0.0
        uy = 0.0
        uz = 0.0
        for iq in range(q):
            ux += f[i, j, k, iq] * c[iq, 0]
            uy += f[i, j, k, iq] * c[iq, 1]
            uz += f[i, j, k, iq] * c[iq, 2]
        ux *= inv_rho
        uy *= inv_rho
        uz *= inv_rho
        u[i, j, k, 0] = ux
        u[i, j, k, 1] = uy
        u[i, j, k, 2] = uz
        
        # Compute u*u for feq
        uu = ux * ux + uy * uy + uz * uz
        
        # Collision with feq computation
        for iq in range(q):
            uc = ux * c[iq, 0] + uy * c[iq, 1] + uz * c[iq, 2]
            feq = w[iq] * local_rho * (1.0 + inv_cs2_val * uc + 0.5 * inv_cs4_val * uc * uc - 0.5 * inv_cs2_val * uu)
            f[i, j, k, iq] = f[i, j, k, iq] - omega * (f[i, j, k, iq] - feq)


@cuda.jit
def feq_kernel_3d(f, feq, rho, u, c, w, inv_cs2_val, inv_cs4_val, nx, ny, nz, q):
    """CUDA kernel for computing equilibrium distribution in 3D."""
    i, j, k = cuda.grid(3)
    if i < nx and j < ny and k < nz:
        ux = u[i, j, k, 0]
        uy = u[i, j, k, 1]
        uz = u[i, j, k, 2]
        uu = ux * ux + uy * uy + uz * uz
        local_rho = rho[i, j, k]
        
        for iq in range(q):
            uc = ux * c[iq, 0] + uy * c[iq, 1] + uz * c[iq, 2]
            feq[i, j, k, iq] = w[iq] * local_rho * (1.0 + inv_cs2_val * uc + 0.5 * inv_cs4_val * uc * uc - 0.5 * inv_cs2_val * uu)


class Lattice:
    def __init__(self, n, stencil: Stencil, use_gpu=True):
        self.stencil = stencil
        self.use_gpu = use_gpu and cuda.is_available()
        self.shape = n
        self.gamma = 1.4
        self._host_dirty = True  # Track if host arrays need sync
        
        # Thread block configuration
        if self.stencil.d == 2:
            self.threads_per_block = (16, 16)
            self.blocks_per_grid = (
                (n[0] + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (n[1] + self.threads_per_block[1] - 1) // self.threads_per_block[1]
            )
        else:  # 3D
            self.threads_per_block = (8, 8, 8)
            self.blocks_per_grid = (
                (n[0] + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (n[1] + self.threads_per_block[1] - 1) // self.threads_per_block[1],
                (n[2] + self.threads_per_block[2] - 1) // self.threads_per_block[2]
            )
        
        if self.use_gpu:
            print("GPU mode enabled - using CUDA")
            # Allocate host arrays (pinned memory for faster transfers)
            self.f = cuda.pinned_array((n + (stencil.q,)), dtype=np.float64)
            self.u = cuda.pinned_array((n + (stencil.d,)), dtype=np.float64)
            self.rho = cuda.pinned_array(n, dtype=np.float64)
            self.f[:] = 0.0
            self.u[:] = 0.0
            self.rho[:] = 1.0
            
            # Create CUDA stream for async operations
            self.stream = cuda.stream()
            
            # Device arrays
            self.d_f = cuda.to_device(self.f, stream=self.stream)
            self.d_f_new = cuda.device_array_like(self.f, stream=self.stream)
            self.d_u = cuda.to_device(self.u, stream=self.stream)
            self.d_rho = cuda.to_device(self.rho, stream=self.stream)
            self.d_c = cuda.to_device(stencil.c, stream=self.stream)
            self.d_w = cuda.to_device(stencil.w, stream=self.stream)
            self.stream.synchronize()
        else:
            print("CPU mode - CUDA not available")
            self.f = np.zeros((n + (stencil.q,)), dtype=np.float64)
            self.u = np.zeros((n + (stencil.d,)), dtype=np.float64)
            self.rho = np.ones(n, dtype=np.float64)

    def init_data(self):
        """Initialize distribution functions with equilibrium."""
        if self.use_gpu:
            # Sync host data to device
            self.d_rho.copy_to_device(self.rho, stream=self.stream)
            self.d_u.copy_to_device(self.u, stream=self.stream)
            
            # Compute feq on GPU
            self._feq_gpu()
            
            # Copy feq to f (swap buffers)
            self.d_f, self.d_f_new = self.d_f_new, self.d_f
            self._host_dirty = True
        else:
            self.f = self.feq()

    def sync_to_host(self):
        """Copy GPU data back to host (lazy sync)."""
        if self.use_gpu and self._host_dirty:
            self.stream.synchronize()
            self.d_f.copy_to_host(self.f, stream=self.stream)
            self.d_u.copy_to_host(self.u, stream=self.stream)
            self.d_rho.copy_to_host(self.rho, stream=self.stream)
            self.stream.synchronize()
            self._host_dirty = False

    def sync_to_device(self):
        """Copy host data to GPU."""
        if self.use_gpu:
            self.d_f.copy_to_device(self.f, stream=self.stream)
            self.d_u.copy_to_device(self.u, stream=self.stream)
            self.d_rho.copy_to_device(self.rho, stream=self.stream)
            self.stream.synchronize()
            self._host_dirty = True

    def print_f(self):
        if self.use_gpu:
            self.sync_to_host()
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
        """Streaming step - move particles to neighboring cells."""
        if self.use_gpu:
            self._streaming_gpu()
        else:
            self._streaming_cpu()

    def _streaming_gpu(self):
        """GPU streaming implementation."""
        if self.stencil.d == 2:
            nx, ny = self.shape
            streaming_kernel_2d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_f_new, self.d_c, nx, ny, self.stencil.q
            )
        else:
            nx, ny, nz = self.shape
            streaming_kernel_3d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_f_new, self.d_c, nx, ny, nz, self.stencil.q
            )
        # Swap buffers (no sync needed - next kernel will wait)
        self.d_f, self.d_f_new = self.d_f_new, self.d_f
        self._host_dirty = True

    def _streaming_cpu(self):
        """CPU streaming implementation (original)."""
        axis = tuple([i for i in range(self.stencil.d)])
        idx = (slice(None),) * self.stencil.d
        for iq in range(self.stencil.q):
            self.f[idx + (iq,)] = np.roll(self.f[idx + (iq,)], self.stencil.c[iq], axis=axis)

    def density(self):
        """Compute density from distribution functions."""
        if self.use_gpu:
            self._density_gpu()
        else:
            self._density_cpu()

    def _density_gpu(self):
        """GPU density computation."""
        if self.stencil.d == 2:
            nx, ny = self.shape
            density_kernel_2d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_rho, nx, ny, self.stencil.q
            )
        else:
            nx, ny, nz = self.shape
            density_kernel_3d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_rho, nx, ny, nz, self.stencil.q
            )
        # No sync - next kernel on same stream will wait

    def _density_cpu(self):
        """CPU density computation (original)."""
        self.rho = np.sum(self.f, axis=self.f.ndim-1)

    def velocity(self):
        """Compute velocity from distribution functions."""
        if self.use_gpu:
            self._velocity_gpu()
        else:
            self._velocity_cpu()

    def _velocity_gpu(self):
        """GPU velocity computation."""
        if self.stencil.d == 2:
            nx, ny = self.shape
            velocity_kernel_2d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_u, self.d_rho, self.d_c, nx, ny, self.stencil.q, self.stencil.d
            )
        else:
            nx, ny, nz = self.shape
            velocity_kernel_3d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_u, self.d_rho, self.d_c, nx, ny, nz, self.stencil.q, self.stencil.d
            )
        # No sync - next kernel on same stream will wait

    def _velocity_cpu(self):
        """CPU velocity computation (original)."""
        idx = (slice(None),) * self.stencil.d
        for i in range(self.stencil.d):
            self.u[idx + (i,)] = np.dot(self.f[idx], self.stencil.c[:, i])/self.rho

    def collision(self, omega):
        """BGK collision operator."""
        if self.use_gpu:
            self._collision_gpu(omega)
        else:
            self._collision_cpu(omega)

    def _collision_gpu(self, omega):
        """GPU collision - fused density, velocity, and collision computation."""
        if self.stencil.d == 2:
            nx, ny = self.shape
            collision_kernel_2d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_rho, self.d_u, self.d_c, self.d_w,
                omega, inv_cs2, inv_cs4, nx, ny, self.stencil.q, self.stencil.d
            )
        else:
            nx, ny, nz = self.shape
            collision_kernel_3d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_rho, self.d_u, self.d_c, self.d_w,
                omega, inv_cs2, inv_cs4, nx, ny, nz, self.stencil.q, self.stencil.d
            )
        # Mark host as dirty - sync only when needed for output
        self._host_dirty = True

    def _collision_cpu(self, omega):
        """CPU collision (original)."""
        self.density()
        self.velocity()
        self.f -= omega * (self.f - self.feq())

    def _feq_gpu(self):
        """GPU equilibrium distribution computation."""
        if self.stencil.d == 2:
            nx, ny = self.shape
            feq_kernel_2d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_f_new, self.d_rho, self.d_u, self.d_c, self.d_w,
                inv_cs2, inv_cs4, nx, ny, self.stencil.q
            )
        else:
            nx, ny, nz = self.shape
            feq_kernel_3d[self.blocks_per_grid, self.threads_per_block, self.stream](
                self.d_f, self.d_f_new, self.d_rho, self.d_u, self.d_c, self.d_w,
                inv_cs2, inv_cs4, nx, ny, nz, self.stencil.q
            )
        # No explicit sync - stream ordering handles dependencies

    def feq(self):
        """Compute equilibrium distribution function (CPU version for compatibility)."""
        idx = (slice(None),) * self.stencil.d
        feq = np.zeros_like(self.f)
        uu = np.sum(self.u**2, axis=self.f.ndim-1)
        for iq, c_i, w_i in zip(range(self.stencil.q), self.stencil.c, self.stencil.w):
            uc = np.dot(self.u[idx], c_i)
            feq[idx + (iq,)] = w_i * self.rho * (1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc**2 - 0.5 * inv_cs2 * uu)
        return feq
