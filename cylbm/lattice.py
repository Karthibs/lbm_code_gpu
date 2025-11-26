import cupy as cp
import numpy as np  # Keep numpy for printing or specific host operations if needed
from cylbm.constants import cs, inv_cs2, inv_cs4
from cylbm.stencil import Stencil

class Lattice:
    def __init__(self, n, stencil: Stencil):
        self.stencil = stencil
        # [Reference 1]: Initialize arrays directly on the Device (GPU)
        self.f = cp.zeros((n + (stencil.q,)), dtype=cp.float64)
        self.u = cp.zeros((n + (stencil.d,)), dtype=cp.float64)
        self.rho = cp.ones(n, dtype=cp.float64)
        self.gamma = 1.4
        
        # Move stencil constants to GPU once to avoid repeated transfers
        # [Reference 2]: cp.asarray moves data from Host to Device
        self.c_gpu = cp.asarray(self.stencil.c)
        self.w_gpu = cp.asarray(self.stencil.w)

    def init_data(self):
        self.f = self.feq()

    def print_f(self):
        # Move to CPU for printing
        f_cpu = cp.asnumpy(self.f) 
        if self.stencil.d == 2:
            for j in range(f_cpu.shape[1]):
                for i in range(f_cpu.shape[0]):
                    ff = [f"{f_cpu[i, j, iq]}" for iq in range(self.stencil.q)]
                    print(f"[{i}, {j}] {ff}")
        elif self.stencil.d == 3:
            for k in range(f_cpu.shape[2]):
                for j in range(f_cpu.shape[1]):
                    for i in range(f_cpu.shape[0]):
                        ff = [f"{f_cpu[i, j, k, iq]}" for iq in range(self.stencil.q)]
                        print(f"[{i}, {j}, {k}] {ff}")

    def streaming(self):
        axis = tuple([i for i in range(self.stencil.d)])
        idx = (slice(None),) * self.stencil.d
        
        # [Reference 3]: cp.roll performs the shift on the GPU. 
        # Note: For very high performance, a custom kernel is preferred over roll, 
        # but cp.roll is the direct, correct equivalent here.
        for iq in range(self.stencil.q):
            # We must use self.stencil.c (CPU) for the shift amount integer, 
            # as roll expects a host scalar for the shift magnitude.
            shift = tuple(self.stencil.c[iq])
            self.f[idx + (iq,)] = cp.roll(self.f[idx + (iq,)], shift, axis=axis)

    def density(self):
        # [Reference 4]: Reduction operation on GPU
        self.rho = cp.sum(self.f, axis=self.f.ndim-1)

    def velocity(self):
        idx = (slice(None),) * self.stencil.d
        # We calculate momentum and divide by rho
        # Using tensordot or manual loop. Keeping loop for clarity with original structure.
        for i in range(self.stencil.d):
            # Use the GPU-resident c_gpu
            self.u[idx + (i,)] = cp.dot(self.f[idx], self.c_gpu[:, i]) / self.rho

    def collision(self, omega):
        self.density()
        self.velocity()
        # Element-wise operations are automatically fused and parallelized by CuPy
        self.f -= omega * (self.f - self.feq())

    def feq(self):
        idx = (slice(None),) * self.stencil.d
        feq = cp.zeros_like(self.f)
        
        # Sum of u^2
        uu = cp.sum(self.u**2, axis=self.f.ndim-1)
        
        # Iterate over GPU resident constants
        # Note: We iterate 'range' on CPU, but operations inside are GPU
        for iq in range(self.stencil.q):
            c_i = self.c_gpu[iq]
            w_i = self.w_gpu[iq]
            
            uc = cp.dot(self.u[idx], c_i)
            feq[idx + (iq,)] = w_i * self.rho * (1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc**2 - 0.5 * inv_cs2 * uu)
        return feq