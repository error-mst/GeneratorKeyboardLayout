import numpy as np
from numpy.typing import NDArray


def optimize_distance():
    pass

def optimize_kbrd_layout(
        kbrd_layout:NDArray[np.int32],
        dist_matrix: NDArray[np.float32],
        trans_weights_matrix: NDArray[np.float32],
        n_iter: int = 15000,
        temperature: float = 0.5,
        cool_rate: float = 0.9998,
        seed: int | None = None,
) -> tuple[NDArray[np.float32], float]:
    rng = np.random.default_rng(seed)

    def score(kbrd_layout: NDArray[np.int32]) -> float:
        exp_dist_matrix = dist_matrix[np.ix_(kbrd_layout, kbrd_layout)]
        return np.float32(np.sum(trans_weights_matrix * exp_dist_matrix))

    n = len(kbrd_layout)
    current = kbrd_layout
    curr_score = score(current)
    best = current
    best_score = curr_score

    for _ in range(n_iter):
        i, j = np.random.choice(n, size=2, replace=False)

        candidate = current.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]

        cand_score = score(candidate)
        diff = cand_score - curr_score

        if diff < 0 or rng.random() < np.exp(-diff / max(temperature, 1e-12)):
            current = candidate
            current_score = cand_score

            if current_score < best_score:
                best_score = current_score
                best = current.copy()
        temperature *= cool_rate

    return best, current_score
