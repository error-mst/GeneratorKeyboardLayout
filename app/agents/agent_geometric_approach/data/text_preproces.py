import re
import numpy as np

from typing import List, Any
from collections import Counter


def extract_alphabet(alphabet: str) -> tuple[set[str], str]:
    alphabet = set(alphabet.lower())
    pattern = "".join(char for char in alphabet)
    pattern = r"[^" + pattern + r"]+"

    return alphabet, pattern


def normalize_text(text: str, pattern: str) -> str:
    return re.sub(pattern, "", text.lower())


def count_char_freq(text: str) -> Counter[str]:
    counter = Counter(text)
    return counter


def create_ngrams(text: str, n: int = 2) -> List[str]:
    return [text[i: i + n] for i in range(len(text) - n + 1)]


def count_trans_freq(text: str) -> Counter[str]:
    bigrams = create_ngrams(text, n=2)
    counter = Counter(bigrams)
    return counter


def create_trans_matrix(counts: Counter[str], alphabet: set[str]) -> tuple[dict[str, int], np.ndarray]:
    n = len(alphabet)
    matrix = np.zeros((n, n), dtype=np.float32)

    # char to index
    ch2idx = {char: i for i, char in enumerate(alphabet)}

    for bigram, count in counts.items():
        i1, i2 = ch2idx[bigram[0]], ch2idx[bigram[1]]
        matrix[i1, i2] = count

    return ch2idx, matrix


def create_prob_matrix(matrix: np.ndarray) -> np.ndarray:
    # create probability matrix
    sum_per_row = matrix.sum(axis=1, keepdims=True, dtype=np.float32)
    sum_per_row[sum_per_row == 0] = 1.0
    matrix = matrix / sum_per_row

    return matrix


def estimate_stat_dist(matrix: np.ndarray) -> np.ndarray:
        P = np.asarray(matrix, dtype=np.float32)

        eigenvalue, eigenvector = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalue - 1.0))
        pi = np.real(eigenvector[:, idx])
        pi = pi / pi.sum()

        return pi


def get_lang_desc(alphabet: str, text: str) -> dict[str, Any]:
    alphabet, pattern = extract_alphabet(alphabet)
    norm_text = normalize_text(text, pattern)
    trans_freq = count_trans_freq(norm_text)
    _, trans_matrix = create_trans_matrix(trans_freq, alphabet)
    prob_matrix = create_prob_matrix(trans_matrix)

    desc = {
        "alphabet": alphabet,
        "pattern": pattern,
        "norm_text": norm_text,
        "char_freq": count_char_freq(norm_text),
        "trans_freq": trans_freq,
        "trans_matrix": trans_matrix,
        "prob_matrix": prob_matrix,
        "stat_dist": estimate_stat_dist(trans_matrix),
    }

    return desc