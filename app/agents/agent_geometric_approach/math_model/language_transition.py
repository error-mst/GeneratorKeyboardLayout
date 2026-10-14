import numpy as np
from numpy.typing import NDArray

def create_trans_weights(prob_matrix: NDArray[np.float32], stat_dist: NDArray[np.float32]) -> NDArray[np.float32]:
    return stat_dist[None, :] * prob_matrix
