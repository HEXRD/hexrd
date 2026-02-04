import numpy as np
from numpy.typing import NDArray

def ge_41rt_inverse_distortion(
    inputs: NDArray[np.float64],
    rhoMax: float,
    params: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...
