import numpy as np
from numpy.typing import NDArray

def encode_alphabet(alphabet: str) -> tuple[dict[str, int], dict[int, str]]:
    key_to_idx = {key: i for i, key in enumerate(alphabet)}
    idx_to_key = {i: key for i, key in enumerate(alphabet)}

    return key_to_idx, idx_to_key

def decode_kbrd_layout(kbrd_layout:NDArray[np.int32], idx_to_key: dict[int, str],
                       size: tuple[int] = (10, 9, 7)) -> list[list[str]]:
    optimized_layout = [[""] * sz for sz in size]
    p = 0
    for i, row in enumerate(optimized_layout):
        for j in range(len(row)):
            optimized_layout[i][j] += idx_to_key[kbrd_layout[p]]
            p += 1

    return optimized_layout

def kbrd_layout_to_array(kbrd_rows: tuple[str], key_to_idx: dict[str, int]) -> NDArray[np.int32]:
    return np.array([key_to_idx[key]
                     for row in kbrd_rows
                     for key in row], dtype=np.int32)


def create_positions(kbrd_rows: tuple[tuple[str, float, float]], key_to_idx: dict[str, int]):
    positions = {}
    positions_array = []

    for row, x, y in kbrd_rows:
        for dx, key in enumerate(row):
            positions[key_to_idx[key]] = (x + dx, y)
            positions_array.append((x + dx, y))

    return positions, positions_array