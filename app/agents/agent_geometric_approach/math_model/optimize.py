import numpy as np
from numpy.typing import NDArray

# Вычисляет ожидаемые расстояния
from .evaluate import compute_expected_distance as score

def optimize_distance():
    pass

def optimize_kbrd_layout(
        kbrd_layout:NDArray[np.int32],
        dist_matrix: NDArray[np.float32],
        trans_weights_matrix,
        n_iter: int = 50000,
        temperature: float = 0.5,
        cool_rate: float = 0.9998,
        seed: int | None = None,
) -> tuple[NDArray[np.int32], float]:
    rng = np.random.default_rng(seed)

    n = len(kbrd_layout)

    current = kbrd_layout
    curr_score = score(current, dist_matrix, trans_weights_matrix)
    best = current
    best_score = curr_score

    for _ in range(n_iter):
        i, j = np.random.choice(n, size=2, replace=False)

        candidate = current.copy()
        # Меняю позиции м/у парой символов
        candidate[i], candidate[j] = candidate[j], candidate[i]

        cand_score = score(candidate, dist_matrix, trans_weights_matrix)
        diff = cand_score - curr_score

        if diff < 0 or rng.random() < np.exp(-diff / max(temperature, 1e-12)):
            current = candidate
            current_score = cand_score

            if current_score < best_score:
                best_score = current_score
                best = current.copy()
        temperature *= cool_rate

    return best, best_score