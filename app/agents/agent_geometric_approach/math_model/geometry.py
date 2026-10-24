import numpy as np
from numpy.typing import NDArray
from typing import Any

def create_dist_matrix(positions: NDArray[np.float32]):
    positions = np.asarray(positions, dtype=np.float32)

    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return distances