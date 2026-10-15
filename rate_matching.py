#!/usr/bin/env python3

import numpy as np
from typing import Tuple


_BG_INFO = {
    1: {"mb": 46, "nb": 68, "kb": 22},
    2: {"mb": 42, "nb": 52, "kb": 10},
}
# start-offset coefficients for each redundancy version
_RV_COEFFS = {
    1: {1: 17, 2: 33, 3: 56},
    2: {1: 13, 2: 25, 3: 43},
}


def build_circular_buffer(
    codeword: np.ndarray,
    K: int,
    Z: int,
    bg_id: int
) -> np.ndarray:
    """
    input:
        codeword : ndarray
        K : int
        Z : int
        bg_id : int
    output:
        buffer : ndarray, same length as codeword, -1 for filler bits
    """
    kb = _BG_INFO[bg_id]["kb"]
    K_sys = kb * Z
    N_filler = K_sys - K

    w = codeword.astype(np.int64).copy()
    if N_filler > 0:
        w[K:K_sys] = -1  # mark filler bits so selection skips them

    return w


def compute_k0(N_cb: int, Z: int, bg_id: int, rv: int = 0) -> int:
    """
    input:
        N_cb : int
        Z : int
        bg_id : int
        rv : int
    output:
        offset : int, circular buffer start
    """
    if rv == 0:
        return 0

    numerator = _RV_COEFFS[bg_id][rv]
    denominator = _BG_INFO[bg_id]["nb"] - 2

    k0 = (numerator * N_cb) // (denominator * Z) * Z
    return k0


def bit_selection(
    w: np.ndarray,
    E: int,
    k0: int
) -> np.ndarray:
    """
    input:
        w : ndarray
        E : int
        k0 : int
    output:
        bits : ndarray, shape (E,)
    """
    N_cb = len(w)
    e = np.zeros(E, dtype=np.uint8)

    k = 0
    j = 0
    while k < E:
        idx = (k0 + j) % N_cb  # wrap around the circular buffer
        if w[idx] != -1:
            e[k] = w[idx]
            k += 1
        j += 1
        if j > N_cb * (E + 1):
            raise RuntimeError("circular buffer cannot supply E bits")

    return e


def bit_interleave(e: np.ndarray, Qm: int) -> np.ndarray:
    """
    input:
        e : ndarray
        Qm : int
    output:
        bits : ndarray, same length as e
    """
    E = len(e)
    if E % Qm != 0:
        raise ValueError(f"length {E} must be divisible by Qm={Qm}")
    # group bits into Qm columns before transposing
    C = E // Qm
    interleaved = e.reshape(C, Qm).T.flatten()
    return interleaved


def bit_deinterleave(f: np.ndarray, Qm: int) -> np.ndarray:
    """
    input:
        f : ndarray
        Qm : int
    output:
        values : ndarray, same length as f
    """
    E = len(f)
    if E % Qm != 0:
        raise ValueError(f"length {E} must be divisible by Qm={Qm}")

    C = E // Qm
    deinterleaved = f.reshape(Qm, C).T.flatten()
    return deinterleaved


def rate_match(
    codeword: np.ndarray,
    E: int,
    Z: int,
    K: int,
    bg_id: int,
    rv: int = 0,
    Qm: int = 2,
    return_interleaved: bool = True
) -> np.ndarray:
    """
    input:
        codeword : ndarray
        E : int
        Z : int
        K : int
        bg_id : int
        rv : int
        Qm : int
        return_interleaved : bool
    output:
        bits : ndarray, shape (E,)
    """
    w = build_circular_buffer(codeword, K, Z, bg_id)
    N_cb = len(w)

    k0 = compute_k0(N_cb, Z, bg_id, rv)

    e = bit_selection(w, E, k0)

    if return_interleaved:
        f = bit_interleave(e, Qm)
        return f
    return e


def rate_recover_llrs(
    rx_llrs: np.ndarray,
    E: int,
    Z: int,
    K: int,
    bg_id: int,
    rv: int = 0,
    Qm: int = 2,
    llr_filler: float = 1000.0
) -> np.ndarray:
    """
    input:
        rx_llrs : ndarray, positive for bit 0
        E : int
        Z : int
        K : int
        bg_id : int
        rv : int
        Qm : int
        llr_filler : float
    output:
        llrs : ndarray, shape (nb * Z,), positive for bit 0
    """
    N_cb = _BG_INFO[bg_id]["nb"] * Z

    e_llrs = bit_deinterleave(rx_llrs, Qm)

    w_llrs = np.zeros(N_cb, dtype=np.float64)  # unobserved bits start with zero confidence

    kb = _BG_INFO[bg_id]["kb"]
    K_sys = kb * Z

    k0 = compute_k0(N_cb, Z, bg_id, rv)
    w_bits = build_circular_buffer(np.zeros(N_cb, dtype=np.int64),
                                   K, Z, bg_id)

    k = 0
    j = 0
    positions = []
    while k < E:
        idx = (k0 + j) % N_cb  # wrap around the circular buffer
        # rebuild the transmit-side selection order
        if w_bits[idx] != -1:
            positions.append(idx)
            k += 1
        j += 1

    for idx, llr in zip(positions, e_llrs):
        w_llrs[idx] += llr  # combine repeated observations within this transmission

    if K_sys > K:
        w_llrs[K:K_sys] = llr_filler  # filler bits are known zeros

    return w_llrs


if __name__ == "__main__":
    # check ordering with values that are easy to trace
    labels = np.arange(8)
    interleaved = bit_interleave(labels, 2)
    np.testing.assert_array_equal(
        interleaved, [0, 2, 4, 6, 1, 3, 5, 7]
    )
    np.testing.assert_array_equal(
        bit_deinterleave(interleaved, 2), labels
    )

    selected = bit_selection(np.array([1, 0, -1, 1]), 6, 0)
    np.testing.assert_array_equal(selected, [1, 0, 1, 1, 0, 1])

    for rv, expected in enumerate([0, 130, 260, 440]):
        assert compute_k0(520, 10, 2, rv) == expected

    rng = np.random.default_rng(42)

    for k in [100, 95]:
        codeword = rng.integers(0, 2, size=520, dtype=np.uint8)
        codeword[k:100] = 0  # filler bits are known zeros

        for rv in range(4):
            for E in [200, 520, 1040]:
                matched = rate_match(
                    codeword, E, 10, k, 2, rv=rv, Qm=2
                )
                llrs = np.where(matched == 0, 4.0, -4.0)
                recovered = rate_recover_llrs(
                    llrs, E, 10, k, 2, rv=rv, Qm=2
                )

                assert len(matched) == E
                assert len(recovered) == 520
                assert np.all(recovered[k:100] == 1000.0)

                if E >= 520:
                    bits = (recovered < 0).astype(np.uint8)
                    np.testing.assert_array_equal(bits, codeword)

        print(f"k={k}: selection and recovery passed")

    print("all checks passed")
