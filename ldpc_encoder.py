#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat
from pathlib import Path


_Z_FAMILIES = [2, 3, 5, 7, 9, 11, 13, 15]
_Z_SETS = {
    2:  [2, 4, 8, 16, 32, 64, 128, 256],
    3:  [3, 6, 12, 24, 48, 96, 192, 384],
    5:  [5, 10, 20, 40, 80, 160, 320],
    7:  [7, 14, 28, 56, 112, 224],
    9:  [9, 18, 36, 72, 144, 288],
    11: [11, 22, 44, 88, 176, 352],
    13: [13, 26, 52, 104, 208],
    15: [15, 30, 60, 120, 240],
}

_BG_INFO = {
    1: {"mb": 46, "nb": 68, "kb": 22, "file": "shift_coeffs_bg_1.mat"},
    2: {"mb": 42, "nb": 52, "kb": 10, "file": "shift_coeffs_bg_2.mat"},
}


def load_shift_coeffs(bg_id: int) -> np.ndarray:
    """
    input:
        bg_id : int
    output:
        coeffs : ndarray, shape (n_nonzero, 10), row, column and eight shifts
    """
    info = _BG_INFO[bg_id]
    mat_path = Path(__file__).parent / info["file"]
    key = f"shift_coeffs_bg_{bg_id}"
    data = loadmat(str(mat_path))[key]
    return np.asarray(data, dtype=np.int64)


def build_shift_matrix(bg_id: int) -> np.ndarray:
    """
    input:
        bg_id : int
    output:
        shift_matrix : ndarray, shape (mb, nb), -1 for missing blocks
    """
    info = _BG_INFO[bg_id]
    mb, nb = info["mb"], info["nb"]
    coeffs = load_shift_coeffs(bg_id)

    shift_matrix = np.full((mb, nb), -1, dtype=np.int64)  # use -1 for missing blocks
    for row in coeffs:
        r, c, v = int(row[0]), int(row[1]), int(row[2])  # use the first lifting family for this helper
        shift_matrix[r, c] = v
    return shift_matrix


def select_base_graph(K: int, R: float = 0.5) -> int:
    """
    input:
        K : int
        R : float
    output:
        bg_id : int, 1 or 2
    """
    if K <= 292:
        return 2
    if K <= 3824 and R <= 0.67:
        return 2
    if R <= 0.25:
        return 2
    return 1


def select_lifting_size(K: int, bg_id: int) -> int:
    """
    input:
        K : int
        bg_id : int
    output:
        Z : int
    """
    kb = _BG_INFO[bg_id]["kb"]
    required = (K + kb - 1) // kb

    all_Z = sorted(z for family in _Z_SETS.values() for z in family)
    for Z in all_Z:
        if Z >= required:
            return Z
    raise ValueError(f"no supported lifting size for K={K}")


def _get_shift_value(coeffs_entry: np.ndarray, Z: int) -> int:
    """
    input:
        coeffs_entry : ndarray
        Z : int
    output:
        shift : int
    """
    for i, family in enumerate(_Z_FAMILIES):
        if Z in _Z_SETS[family]:
            return int(coeffs_entry[2 + i]) # skip the row and column fields
    raise ValueError(f"Z={Z} is not in a supported lifting set")


def construct_H(coeffs: np.ndarray, mb: int, nb: int, Z: int) -> np.ndarray:
    """
    input:
        coeffs : ndarray
        mb : int
        nb : int
        Z : int
    output:
        matrix : ndarray, shape (mb * Z, nb * Z)
    """
    H = np.zeros((mb * Z, nb * Z), dtype=np.uint8)

    for row in coeffs:
        r, c = int(row[0]), int(row[1])
        V = _get_shift_value(row, Z)

        sub = np.zeros((Z, Z), dtype=np.uint8)
        for i in range(Z):
            sub[i, (i + V) % Z] = 1  # expand each shift into a circulant block

        H[r * Z:(r + 1) * Z, c * Z:(c + 1) * Z] = sub

    return H


def gf2_gaussian_elimination(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    input:
        H : ndarray
        x : ndarray
    output:
        parity_bits : ndarray, shape (H.shape[1] - len(x),)
    """
    K_sys = len(x)
    A = H[:, K_sys:] # parity columns
    b = (H[:, :K_sys] @ x) % 2  # move the known bits to the right-hand side

    M = A.shape[1]
    m = A.shape[0]

    aug = np.hstack([A.astype(np.uint8), b.reshape(-1, 1).astype(np.uint8)]) % 2

    row = 0
    col = 0
    pivots = []
    while row < m and col < M:
        pivot_row = None
        for rr in range(row, m):
            if aug[rr, col] == 1:
                pivot_row = rr
                break
        if pivot_row is None:
            col += 1
            continue
        if pivot_row != row:
            aug[[row, pivot_row]] = aug[[pivot_row, row]]  # put the pivot row in place
        for rr in range(row + 1, m):
            if aug[rr, col] == 1:
                aug[rr] = (aug[rr] + aug[row]) % 2  # eliminate the bit below the pivot
        pivots.append(col)
        row += 1
        col += 1

    rank = len(pivots)
    if rank < M:
        print(f"warning: parity matrix rank is {rank}, expected {M}")

    p = np.zeros(M, dtype=np.uint8)
    for i in range(rank - 1, -1, -1):
        pc = pivots[i]
        val = aug[i, M]
        for j in range(pc + 1, M):
            if aug[i, j] == 1:
                val ^= p[j]
        p[pc] = val

    return p


def ldpc_encode(data_bits: np.ndarray, bg_id=None,
                R: float = 0.5, return_Z: bool = False):
    """
    input:
        data_bits : ndarray
        bg_id : int | none
        R : float
        return_Z : bool
    output:
        codeword : ndarray, including filler bits, full graph codeword with filler bits
        Z : int, included when return_Z is true
    """
    K = len(data_bits)
    if bg_id is None:
        bg_id = select_base_graph(K, R)

    info = _BG_INFO[bg_id]
    mb, nb, kb = info["mb"], info["nb"], info["kb"]

    Z = select_lifting_size(K, bg_id)
    K_sys = kb * Z
    N = nb * Z
    M = N - K_sys

    coeffs = load_shift_coeffs(bg_id)
    H = construct_H(coeffs, mb, nb, Z)

    x = np.zeros(K_sys, dtype=np.uint8)
    x[:K] = np.asarray(data_bits, dtype=np.uint8)  # leave filler bits at zero

    p = gf2_gaussian_elimination(H, x)

    codeword = np.concatenate([x, p])

    syndrome = (H @ codeword) % 2  # check the encoded parity constraints
    if np.any(syndrome != 0):
        raise RuntimeError( f"encoding failed: {np.count_nonzero(syndrome)} parity checks failed"
)

    if return_Z:
        return codeword, Z
    return codeword


def extract_info_bits(codeword: np.ndarray, K: int) -> np.ndarray:
    """
    input:
        codeword : ndarray
        K : int
    output:
        bits : ndarray, shape (K,)
    """
    return codeword[:K].copy()


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # check both base graphs and filler bits
    for k, bg, expected_z in [(100, 2, 10), (95, 2, 10), (44, 1, 2)]:
        bits = rng.integers(0, 2, size=k, dtype=np.uint8)
        codeword, z = ldpc_encode(bits, bg_id=bg, return_Z=True)

        assert z == expected_z
        np.testing.assert_array_equal(codeword[:k], bits)

        k_sys = _BG_INFO[bg]["kb"] * z
        assert np.all(codeword[k:k_sys] == 0)

        print(f"bg{bg}: k={k}, z={z}, n={len(codeword)} passed")

    # check the base graph selection boundaries
    cases = [
        (292, 0.9, 2),
        (293, 0.9, 1),
        (3824, 0.67, 2),
        (3825, 0.67, 1),
        (4000, 0.25, 2),
        (4000, 0.26, 1),
    ]
    for k, rate, expected_bg in cases:
        assert select_base_graph(k, rate) == expected_bg

    print("all checks passed")
