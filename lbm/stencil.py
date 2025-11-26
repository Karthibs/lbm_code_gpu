#Defines the discrete velocity sets. It currently supports D2Q9 (2D, 9 velocities) and D3Q19 (3D, 19 velocities).
#the directions in which fluid particles can move on a lattice grid.

# import numpy as np
import cupy as np



class Stencil:
    def __init__(self, d: int, q: int):
        self._d = d # number of spatial dimensions
        self._q = q # number of discrete velocity 
        self._w = None # probability weights for each discrete velocity
        self._c = None # discrete velocity directions

        if self.d == 2 and self.q == 9:
            self.__d2q9()
        elif self.d == 3 and self.q == 19:
            self.__d3q19()
        else:
            raise Exception(f'Invalid choice of DdQq: "D{self.d}Q{self.q}"')

    @property
    def d(self):
        return self._d

    @property
    def q(self):
        return self._q

    @property
    def w(self):
        return self._w

    @property
    def c(self):
        return self._c

    # Index 0: Rest particle [0,0] with weight 4/9
    # Indices 1-4: Cardinal directions (→↑←↓) with weight 1/9 each
    # Indices 5-8: Diagonal directions with weight 1/36 each
    def __d2q9(self):
        self._w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
        self._c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)

    # Index 0: Rest particle [0,0,0] with weight 1/3
    # Indices 1-6: Face-centered directions (±x, ±y, ±z) with weight 1/18
    # Indices 7-18: Edge-centered directions with weight 1/36
    def __d3q19(self):
        self._w = np.array([1/3,
                            1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                            1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
                            1/36, 1/36, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
        self._c = np.array([[0, 0, 0],
                            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                            [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
                            [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1]], dtype=np.int32)



# Why These Weights?
# The weights w are carefully chosen to satisfy:
# Isotropy — the lattice behaves the same in all directions
# Conservation laws — mass, momentum, and energy are conserved
# Correct recovery of the Navier-Stokes equations in the macroscopic limit

#Porting notes
# The Stencil class is a Python object — Numba njit / CUDA kernels cannot accept arbitrary Python objects. Two ways:
# Keep Stencil on host and extract w and c arrays (NumPy arrays). Pass those arrays to kernels as plain arrays.
# If you want a typed container, use Numba typed Dict or a small C-like struct replacement — but typically simpler to pass w (1D float array) and c (2D int array).
# Ensure c dtype is int32 and contiguous (C-order). On GPU, int32 is fine.