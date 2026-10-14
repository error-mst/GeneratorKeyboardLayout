import numpy as np
from math_model.geometry import create_dist_matrix
from math_model.language_transition import create_trans_weights
from math_model.evaluate import compute_expected_distance
from math_model.optimize import optimize_kbrd_layout

positions = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 1.0],
    [3.0, 2.0],
    [3.0, 1.0],
    [5.0, 4.0],
])

P = np.array([
    [0.0, 0.8, 0.2],
    [0.5, 0.0, 0.5],
    [0.4, 0.6, 0.0],
])

pi = np.array([0.4, 0.4, 0.2, 0.4])
layout = np.array([0, 1, 2, 3, 4, 5])

dist_matrix = create_dist_matrix(positions)
trans_weights_matrix = create_trans_weights(P, pi)
exp_dist_matrix = compute_expected_distance(layout, dist_matrix, trans_weights_matrix)

result = {
    'dist_matrix': dist_matrix,
    'trans_weights_matrix': trans_weights_matrix,
    'exp_dist_matrix': exp_dist_matrix,
}

for key, value in result.items():
    print(f"\n---------\n{key}", value, "\n---------\n", sep="\n")

best, score = optimize_kbrd_layout(layout, dist_matrix, trans_weights_matrix)
print(f"\nBest: {best}\nScore: {score}")