#!/usr/bin/env python3

import numpy as np

from typing import Optional

from ldpc_encoder import (
    load_shift_coeffs,
    _get_shift_value,
    _BG_INFO,
    select_base_graph,
)


def build_edge_list(bg_id: int, Z: int):
    """
    input:
        bg_id : int
        Z : int
    output:
        chk, var : ndarray, edge node indices
        starts, deg : ndarray, check node groups
        n_chk, N : int, check and variable node counts
    """
    info = _BG_INFO[bg_id]
    mb, nb = info["mb"], info["nb"]
    coeffs = load_shift_coeffs(bg_id)

    chk_parts = []
    var_parts = []
    i_vec = np.arange(Z, dtype=np.int64)

    for row in coeffs:
        r = int(row[0])
        c = int(row[1])
        V = _get_shift_value(row, Z)

        chk_parts.append(r * Z + i_vec)
        var_parts.append(c * Z + ((i_vec + V) % Z))

    chk = np.concatenate(chk_parts)
    var = np.concatenate(var_parts)

    order = np.argsort(chk, kind="stable")  # keep edges from the same check node together
    chk = chk[order]
    var = var[order]

    n_chk = mb * Z
    N = nb * Z
    deg = np.bincount(chk, minlength=n_chk).astype(np.int64)
    if np.any(deg == 0):
        raise RuntimeError("found a check node with no edges")

    starts = np.concatenate([[0], np.cumsum(deg)[:-1]]).astype(np.int64)

    return chk, var, starts, deg, n_chk, N


def _check_update_minsum(v2c: np.ndarray, starts: np.ndarray,
                         deg: np.ndarray, alpha: float) -> np.ndarray:
    """
    input:
        v2c : ndarray
        starts : ndarray
        deg : ndarray
        alpha : float
    output:
        c2v : ndarray, shape (len(v2c),)
    """
    signs = np.where(v2c >= 0.0, 1.0, -1.0)
    absv = np.abs(v2c)

    min1 = np.minimum.reduceat(absv, starts)
    min1_e = np.repeat(min1, deg)

    is_min = absv <= min1_e
    cnt_min = np.add.reduceat(is_min.astype(np.int64), starts)
    cnt_min_e = np.repeat(cnt_min, deg)

    absv2 = np.where(is_min, np.inf, absv)  # exclude minimum edges when finding the second minimum
    min2 = np.minimum.reduceat(absv2, starts)
    min2_e = np.repeat(min2, deg)

    min_other = np.where(
        absv > min1_e,
        min1_e,
        np.where(cnt_min_e >= 2, min1_e, min2_e),
    )

    sign_prod = np.multiply.reduceat(signs, starts)
    sign_prod_e = np.repeat(sign_prod, deg)
    sign_other = sign_prod_e * signs  # remove the current edge from the sign product

    c2v = alpha * sign_other * min_other
    return c2v


def _check_update_sumproduct(v2c: np.ndarray, starts: np.ndarray,
                             deg: np.ndarray) -> np.ndarray:
    """
    input:
        v2c : ndarray
        starts : ndarray
        deg : ndarray
    output:
        c2v : ndarray, shape (len(v2c),)
    """
    t = np.tanh(v2c / 2.0)
    is_zero = t == 0.0
    t_safe = np.where(is_zero, 1.0, t)

    prod_nonzero = np.multiply.reduceat(t_safe, starts)
    prod_nonzero_e = np.repeat(prod_nonzero, deg)
    zero_count = np.add.reduceat(is_zero.astype(np.int64), starts)
    zero_count_e = np.repeat(zero_count, deg)

    # exclude the current edge without losing a single zero input
    t_other = prod_nonzero_e / t_safe
    t_other = np.where(
        zero_count_e == 0,
        t_other,
        np.where((zero_count_e == 1) & is_zero, prod_nonzero_e, 0.0),
    )
    t_other = np.clip(t_other, -1.0 + 1e-12, 1.0 - 1e-12)

    c2v = 2.0 * np.arctanh(t_other)
    return c2v


def _hard_decision(total_llr: np.ndarray) -> np.ndarray:
    """
    input:
        total_llr : ndarray
    output:
        bits : ndarray, 1 where total_llr is negative
    """
    return (total_llr < 0.0).astype(np.uint8)


def syndrome_check(bits: np.ndarray, Z: int, bg_id: int) -> int:
    """
    input:
        bits : ndarray
        Z : int
        bg_id : int
    output:
        count : int, failed parity checks
    """
    chk, var, starts, deg, n_chk, N = build_edge_list(bg_id, Z)
    bits = np.asarray(bits, dtype=np.uint8)
    syn = np.bitwise_xor.reduceat(bits[var], starts)
    return int(np.sum(syn != 0))


def ldpc_decode(
    llr: np.ndarray,
    Z: int,
    bg_id: int,
    K: Optional[int] = None,
    max_iter: int = 25,
    alpha: float = 0.75,
    algorithm: str = "minsum",
    early_stop: bool = True,
    return_info: bool = False,
    return_iter: bool = False,
):
    """
    input:
        llr : ndarray, positive for bit 0
        Z : int
        bg_id : int
        K : int | none
        max_iter : int
        alpha : float
        algorithm : str
        early_stop : bool
        return_info : bool
        return_iter : bool
    output:
        decoded : ndarray, full codeword or first K bits
        iters : int, included when return_iter is true
    """
    if algorithm not in ("minsum", "sumproduct"):
        raise ValueError(f"unknown algorithm: {algorithm}")

    llr = np.asarray(llr, dtype=np.float64).ravel()
    chk, var, starts, deg, n_chk, N = build_edge_list(bg_id, Z)
    if llr.shape[0] != N:
        raise ValueError(
            f"llr length {llr.shape[0]} != codeword length N={N} (nb={_BG_INFO[bg_id]['nb']}, Z={Z})"
        )

    L = llr.copy()  # keep the channel prior separate from the messages
    c2v = np.zeros(var.shape[0], dtype=np.float64)

    decoded = _hard_decision(L)
    it_used = 0

    for it in range(max_iter):
        it_used = it + 1

        sum_c2v = np.bincount(var, weights=c2v, minlength=N)
        v2c = L[var] + sum_c2v[var] - c2v  # exclude the message on the current edge

        if algorithm == "minsum":
            c2v = _check_update_minsum(v2c, starts, deg, alpha)
        else:
            c2v = _check_update_sumproduct(v2c, starts, deg)

        sum_c2v = np.bincount(var, weights=c2v, minlength=N)
        total = L + sum_c2v
        decoded = _hard_decision(total)

        if early_stop:
            syn = np.bitwise_xor.reduceat(decoded[var], starts)
            if not np.any(syn):  # stop when all parity checks pass
                break

    if return_info:
        if K is None:
            raise ValueError("K is required when return_info=True")
        decoded = decoded[:K].copy()

    if return_iter:
        return decoded, it_used
    return decoded



def bits_to_llr(bits: np.ndarray, magnitude: float = 4.0) -> np.ndarray:
    """
    input:
        bits : ndarray
        magnitude : float
    output:
        llrs : ndarray, same shape as bits, positive for bit 0
    """
    bits = np.asarray(bits)
    return np.where(bits == 1, -magnitude, +magnitude).astype(np.float64)


if __name__ == "__main__":
    from ldpc_encoder import ldpc_encode
    from rate_matching import rate_match, rate_recover_llrs

    # a zero input can still receive a nonzero check message
    np.testing.assert_allclose(
        _check_update_sumproduct(
            np.array([0.0, 2.0]), np.array([0]), np.array([2])
        ),
        [2.0, 0.0], rtol=0, atol=1e-12
    )

    rng = np.random.default_rng(2024)
    K = 100
    data = rng.integers(0, 2, size=K).astype(np.uint8)
    codeword, Z = ldpc_encode(data, return_Z=True)
    bg_id = select_base_graph(K, 0.5)
    assert syndrome_check(codeword, Z, bg_id) == 0

    clean_llrs = bits_to_llr(codeword, magnitude=4.0)
    corrupted_llrs = clean_llrs.copy()
    error_positions = rng.choice(len(codeword), size=8, replace=False)
    corrupted_llrs[error_positions] *= -1.0

    E = 200
    matched = rate_match(codeword, E=E, Z=Z, K=K, bg_id=bg_id, rv=0, Qm=2)
    recovered_llrs = rate_recover_llrs(
        bits_to_llr(matched, magnitude=5.0),
        E=E, Z=Z, K=K, bg_id=bg_id, rv=0, Qm=2
    )

    for algorithm in ("minsum", "sumproduct"):
        for llrs in (clean_llrs, corrupted_llrs, recovered_llrs):
            decoded, iters = ldpc_decode(
                llrs, Z, bg_id, K=K, max_iter=50,
                algorithm=algorithm, return_info=True, return_iter=True
            )
            np.testing.assert_array_equal(decoded, data)
            assert 1 <= iters <= 50
        print(f"{algorithm} checks passed")
