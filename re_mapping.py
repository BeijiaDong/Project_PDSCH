#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from dmrs import generate_dmrs_symbols


@dataclass
class REFrameConfig:
    n_prb: int = 20
    n_sc_per_prb: int = 12
    n_sym_per_slot: int = 14
    dmrs_type: int = 1 # only type 1 is implemented
    dmrs_sym_positions: Tuple[int, ...] = (2, 11)
    dmrs_cdm_group: int = 0 # mapping uses cdm group 0
    fft_size: int = 1024
    scs: int = 30
    pdsch_start_sym: int = 2
    pdsch_num_sym: int = 12

    @property
    def n_sc(self) -> int:
        return self.n_prb * self.n_sc_per_prb

    @property
    def resource_grid_shape(self) -> Tuple[int, int]:
        return (self.n_sc, self.n_sym_per_slot)


def get_dmrs_re_positions(config: REFrameConfig) -> np.ndarray:
    """
    input:
        config : REFrameConfig
    output:
        positions : ndarray, shape (n_dmrs, 2), columns (sc, sym)
    """
    dmrs_sym_positions = config.dmrs_sym_positions

    dmrs_sc_offsets = np.array([0, 2, 4, 6, 8, 10])  # offsets within each prb

    dmrs_positions = []
    for sym_idx in dmrs_sym_positions:  # keep pilot ordering consistent with the generator
        for prb_idx in range(config.n_prb):
            base_sc = prb_idx * config.n_sc_per_prb
            for offset in dmrs_sc_offsets:
                dmrs_positions.append([base_sc + offset, sym_idx])


    return np.array(dmrs_positions, dtype=int)


def get_pdsch_re_positions(config: REFrameConfig) -> np.ndarray:
    """
    input:
        config : REFrameConfig
    output:
        positions : ndarray, shape (n_pdsch, 2), columns (sc, sym)
    """
    n_sc = config.n_sc
    start_sym = config.pdsch_start_sym
    end_sym = start_sym + config.pdsch_num_sym

    dmrs_full = get_dmrs_re_positions(config)
    dmrs_set = set((int(row[0]), int(row[1])) for row in dmrs_full)  # use a set to skip pilot positions

    pdsch_positions = []
    # scan subcarriers in order within each symbol
    for sym_idx in range(start_sym, end_sym):
        for sc_idx in range(n_sc):
            if (sc_idx, sym_idx) not in dmrs_set:
                pdsch_positions.append([sc_idx, sym_idx])

    return np.array(pdsch_positions, dtype=int)


def resource_grid_mapping(
    pdsch_symbols: np.ndarray,
    dmrs_symbols: np.ndarray,
    config: Optional[REFrameConfig] = None
) -> np.ndarray:
    """
    input:
        pdsch_symbols : ndarray
        dmrs_symbols : ndarray
        config : REFrameConfig | none
    output:
        grid : ndarray, shape (n_sc, n_sym)
    """
    if config is None:
        config = REFrameConfig()

    n_sc = config.n_sc
    n_sym = config.n_sym_per_slot

    dmrs_positions = get_dmrs_re_positions(config)
    pdsch_positions = get_pdsch_re_positions(config)

    dmrs = np.asarray(dmrs_symbols, dtype=complex).ravel()
    pdsch = np.asarray(pdsch_symbols, dtype=complex).ravel()

    resource_grid = np.zeros((n_sc, n_sym), dtype=complex)  # rows are subcarriers and columns are ofdm symbols

    for idx, (sc, sym) in enumerate(dmrs_positions):
        resource_grid[sc, sym] = dmrs[idx]

    for idx, (sc, sym) in enumerate(pdsch_positions):
        resource_grid[sc, sym] = pdsch[idx]

    return resource_grid


def resource_grid_demapping(
    resource_grid: np.ndarray,
    config: Optional[REFrameConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    input:
        resource_grid : ndarray
        config : REFrameConfig | none
    output:
        pdsch, dmrs : ndarray, grouped by symbol
    """
    if config is None:
        config = REFrameConfig()

    dmrs_positions = get_dmrs_re_positions(config)
    pdsch_positions = get_pdsch_re_positions(config)

    dmrs_rx = np.zeros(dmrs_positions.shape[0], dtype=complex)
    # use the same coordinate order as the mapper
    for idx, (sc, sym) in enumerate(dmrs_positions):
        dmrs_rx[idx] = resource_grid[sc, sym]

    pdsch_rx = np.zeros(pdsch_positions.shape[0], dtype=complex)
    for idx, (sc, sym) in enumerate(pdsch_positions):
        pdsch_rx[idx] = resource_grid[sc, sym]

    return pdsch_rx, dmrs_rx

if __name__ == "__main__":
    for n_prb in [2, 20]:
        config = REFrameConfig(n_prb=n_prb)
        pdsch_pos = get_pdsch_re_positions(config)
        dmrs_pos = get_dmrs_re_positions(config)

        pdsch = np.arange(len(pdsch_pos), dtype=complex)
        dmrs = np.arange(len(dmrs_pos), dtype=complex) + 10000j

        grid = resource_grid_mapping(pdsch, dmrs, config)
        pdsch_rx, dmrs_rx = resource_grid_demapping(grid, config)

        np.testing.assert_array_equal(pdsch_rx, pdsch)
        np.testing.assert_array_equal(dmrs_rx, dmrs)
        assert grid.shape == config.resource_grid_shape

        # check a few known positions as well as the roundtrip
        assert grid[0, 2] == dmrs[0]
        assert grid[3, 2] == pdsch[1]
        assert grid[0, 3] == pdsch[6 * n_prb]
        assert grid[0, 11] == dmrs[6 * n_prb]
        assert np.all(grid[:, :2] == 0)

        print(f"prb={n_prb}: mapping and demapping passed")

    print("all checks passed")