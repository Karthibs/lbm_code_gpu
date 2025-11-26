#Defines physical constants like the speed of sound

import numpy as np
# import cupy as cp

cs = np.float32(1.0 / np.sqrt(3))
cs2 = np.float32(1.0/3.0)
cs4 = np.float32(1.0/9.0)
inv_cs = np.float32(np.sqrt(3))

# used in lattice.py/feq()
inv_cs2 = np.float32(3)
inv_cs4 = np.float32(9)


#for numba chnage to float32 bcz gpus operate faster at single precision. 
