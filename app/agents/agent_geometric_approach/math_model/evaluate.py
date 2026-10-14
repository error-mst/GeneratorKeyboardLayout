import numpy as np
from numpy.typing import NDArray

def compute_expected_distance(kbrd_layout: NDArray[np.int32], dist_matrix: NDArray[np.float32],
                              trans_weights_matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    exp_dist_matrix = dist_matrix[np.ix_(kbrd_layout, kbrd_layout)]
    return np.float32(np.sum(trans_weights_matrix * exp_dist_matrix))