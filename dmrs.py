#!/usr/bin/env python3

import numpy as np
from gold_sequence import gold_sequence
from typing import Tuple


def dmrs_c_init(
    n_slot: int,
    l: int,
    N_ID: int = 1,
    n_SCID: int = 0,
    N_symb_slot: int = 14,
) -> int:
    """
    input:
        n_slot : int
        l : int, symbol index within a slot
        N_ID : int
        n_SCID : int
        N_symb_slot : int
        mu : int, unused
    output:
        c_init : int, 31-bit seed
    """
    term1 = N_symb_slot * n_slot + l + 1
    term2 = 2 * N_ID + 1
    c_init = (2**17) * term1 * term2 + 2 * N_ID + n_SCID
    c_init = c_init % (2**31)  # keep the seed within 31 bits
    return c_init


def dmrs_symbols_from_seed(c_init: int, length: int) -> np.ndarray:
    """
    input:
        c_init : int
        length : int
    output:
        symbols : ndarray, shape (length,)
    """
    bits = gold_sequence(c_init, 2 * length)
    bits = np.array(bits, dtype=int)

    symbols = np.zeros(length, dtype=complex)

    for n in range(length):
        real_part = 1.0 - 2.0 * bits[2 * n]
        imag_part = 1.0 - 2.0 * bits[2 * n + 1]
        symbols[n] = (real_part + 1j * imag_part) / np.sqrt(2)

    return symbols


def generate_dmrs_symbols(
    N_ID: int = 1,
    n_slot: int = 0,
    n_SCID: int = 0,
    dmrs_sym_positions: Tuple[int, ...] = (2, 11),
    n_prb: int = 20,
    dmrs_sc_per_prb: int = 6
) -> np.ndarray:
    """
    input:
        N_ID : int
        n_slot : int
        n_SCID : int
        dmrs_sym_positions : tuple[int, ...]
        n_prb : int
        dmrs_sc_per_prb : int
    output:
        dmrs : ndarray, grouped by symbol in the given order
    """
    n_per_symbol = n_prb * dmrs_sc_per_prb
    dmrs = np.empty(len(dmrs_sym_positions) * n_per_symbol, dtype=complex)

    # store each symbol's pilots in one contiguous block
    for i, l in enumerate(dmrs_sym_positions):
        c_init = dmrs_c_init(n_slot=n_slot, l=l, N_ID=N_ID, n_SCID=n_SCID)
        start = i * n_per_symbol
        dmrs[start:start + n_per_symbol] = dmrs_symbols_from_seed(
            c_init, n_per_symbol
        )

    return dmrs


if __name__ == "__main__":
    assert dmrs_c_init(n_slot=0, l=2, N_ID=1) == 1179650
    assert dmrs_c_init(n_slot=19, l=13, N_ID=65535, n_SCID=1) == 2110914559

    dmrs = generate_dmrs_symbols(n_prb=2)
    assert dmrs.shape == (24,)
    assert np.allclose(np.abs(dmrs), 1.0)

    # check fixed prefixes for symbols 2 and 11
    expected = np.array([
        1 + 1j, 1 + 1j, -1 - 1j, -1 + 1j,
        1 + 1j, -1 + 1j, 1 + 1j, 1 - 1j
    ]) / np.sqrt(2)
    np.testing.assert_allclose(
        dmrs[[0, 1, 2, 3, 12, 13, 14, 15]],
        expected,
        rtol=0,
        atol=1e-12
    )

    print("dmrs checks passed")