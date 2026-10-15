#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ChannelConfig:
    fft_size: int = 1024
    scs_khz: float = 30.0
    n_prb: int = 20
    n_sc: int = 240
    cp_lengths: Tuple[int, ...] = (88,) + (72,) * 13
    dmrs_sc_per_prb: int = 6

    @property
    def sampling_rate(self) -> float:
        return self.fft_size * self.scs_khz * 1000.0

    @property
    def sample_period_ns(self) -> float:
        return 1e9 / self.sampling_rate

# simplified static tap profile: delay in ns, power in db
TDL_C_300NS_TAPS: List[Tuple[float, float]] = [
    (0.0,    -4.4),
    (65.0,   -1.2),
    (70.0,   -3.5),
    (190.0,  -5.2),
    (195.0,  -2.5),
    (200.0,  0.0),
    (210.0,  -1.5),
    (295.0,  -5.4),
    (300.0,  -7.0),
    (310.0,  -2.3),
    (340.0,  -5.6),
    (435.0,  -9.1),
    (515.0,  -11.2),
]

DELAY_SCALE: float = 1.0


def generate_tdl_channel(
    n_taps: int = 13,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    input:
        n_taps : int
        rng : RandomState | none
    output:
        cir : ndarray, unit total power
    """
    if rng is None:
        rng = np.random.RandomState()

    taps_info = TDL_C_300NS_TAPS[:n_taps]
    cir = np.zeros(n_taps, dtype=complex)

    for i, (delay_ns, power_dB) in enumerate(taps_info):
        power_linear = 10 ** (power_dB / 10.0)
        amplitude = np.sqrt(power_linear)

        real_part = rng.randn()
        imag_part = rng.randn()
        cir[i] = amplitude * (real_part + 1j * imag_part) / np.sqrt(2.0)

    total_power = np.sum(np.abs(cir) ** 2)
    if total_power > 1e-20:
        cir = cir / np.sqrt(total_power)  # normalize the power of this channel realization

    return cir


def apply_channel(
    tx_signal: np.ndarray,
    cir: np.ndarray,
    config: Optional[ChannelConfig] = None
) -> np.ndarray:
    """
    input:
        tx_signal : ndarray
        cir : ndarray
        config : ChannelConfig | none
    output:
        rx_signal : ndarray, same length as tx_signal
    """
    if config is None:
        config = ChannelConfig()

    ts_ns = config.sample_period_ns

    n_taps = len(cir)
    taps_info = TDL_C_300NS_TAPS[:n_taps]
    delays_ns = np.array([t[0] for t in taps_info]) * DELAY_SCALE

    delays_samples = np.round(delays_ns / ts_ns).astype(int)  # round delays to the sampling grid

    N = len(tx_signal)
    rx_signal = np.zeros(N, dtype=complex)

    for i in range(n_taps):
        d = delays_samples[i]
        if d == 0:
            rx_signal += cir[i] * tx_signal
        else:
            rx_signal[d:] += cir[i] * tx_signal[:N - d]

    return rx_signal




DMRS_SC_OFFSETS_TYPE1: Tuple[int, ...] = (0, 2, 4, 6, 8, 10)


def get_dmrs_sc_indices(
    n_prb: int = 20,
    dmrs_sc_per_prb: int = 6
) -> np.ndarray:
    """
    input:
        n_prb : int
        dmrs_sc_per_prb : int
    output:
        indices : ndarray, shape (n_prb * dmrs_sc_per_prb,)
    """
    dmrs_local = np.asarray(DMRS_SC_OFFSETS_TYPE1[:dmrs_sc_per_prb])
    sc_indices = np.concatenate([
        prb * 12 + dmrs_local for prb in range(n_prb)
    ])
    return sc_indices.astype(int)


def ls_channel_estimate_dmrs(
    dmrs_rx: np.ndarray,
    dmrs_tx: np.ndarray,
) -> np.ndarray:
    """
    input:
        dmrs_rx : ndarray
        dmrs_tx : ndarray, unit-power pilot symbols
    output:
        channel : ndarray, same shape as dmrs_rx
    """
    h_est = dmrs_rx * np.conj(dmrs_tx)  # the pilots have unit power
    return h_est


def linear_interpolate_channel(
    h_est_dmrs: np.ndarray,
    dmrs_sc_indices: np.ndarray,
    n_sc: int = 240
) -> np.ndarray:
    """
    input:
        h_est_dmrs : ndarray
        dmrs_sc_indices : ndarray
        n_sc : int
    output:
        channel : ndarray, shape (n_sc,)
    """
    sort_idx = np.argsort(dmrs_sc_indices)
    sc_pos = dmrs_sc_indices[sort_idx].astype(float)
    h_val = np.asarray(h_est_dmrs[sort_idx], dtype=np.complex128)

    sc_all = np.arange(n_sc, dtype=float)

    real_part = np.asarray(np.interp(sc_all, sc_pos, h_val.real), dtype=float)  # interpolate the real and imaginary parts separately
    imag_part = np.asarray(np.interp(sc_all, sc_pos, h_val.imag), dtype=float)
    h_est_full = (real_part + 1j * imag_part).astype(np.complex128)

    return h_est_full


def channel_estimate_per_symbol(
    dmrs_rx_per_symbol: np.ndarray,
    dmrs_tx_per_symbol: np.ndarray,
    dmrs_sc_indices: np.ndarray,
    n_sc: int = 240
) -> np.ndarray:
    """
    input:
        dmrs_rx_per_symbol : ndarray
        dmrs_tx_per_symbol : ndarray
        dmrs_sc_indices : ndarray
        n_sc : int
    output:
        channel : ndarray, shape (n_sc,)
    """
    h_ls = ls_channel_estimate_dmrs(dmrs_rx_per_symbol, dmrs_tx_per_symbol)
    h_est = linear_interpolate_channel(h_ls, dmrs_sc_indices, n_sc)
    return h_est


def channel_estimate_full(
    dmrs_rx_all: np.ndarray,
    dmrs_tx_all: np.ndarray,
    dmrs_sym_positions: Tuple[int, ...] = (2, 11),
    n_prb: int = 20,
    dmrs_sc_per_prb: int = 6,
    n_sc: int = 240,
    n_sym_slot: int = 14
) -> np.ndarray:
    """
    input:
        dmrs_rx_all : ndarray
        dmrs_tx_all : ndarray
        dmrs_sym_positions : tuple[int, ...]
        n_prb : int
        dmrs_sc_per_prb : int
        n_sc : int
        n_sym_slot : int, unused
    output:
        channel : ndarray, shape (n_sc,), averaged across pilot symbols
    """
    dmrs_sc_indices = get_dmrs_sc_indices(n_prb, dmrs_sc_per_prb)
    n_per_sym = n_prb * dmrs_sc_per_prb

    h_accum = np.zeros(n_sc, dtype=complex)
    n_sym_count = 0

    for sym_idx in range(len(dmrs_sym_positions)):
        start = sym_idx * n_per_sym
        end = start + n_per_sym
        dmrs_rx_sym = dmrs_rx_all[start:end]
        dmrs_tx_sym = dmrs_tx_all[start:end]

        h_sym = channel_estimate_per_symbol(
            dmrs_rx_sym, dmrs_tx_sym, dmrs_sc_indices, n_sc
        )
        h_accum += h_sym
        n_sym_count += 1

    h_est_avg = h_accum / n_sym_count  # average the pilot-symbol estimates
    return h_est_avg


def zf_equalize(
    pdsch_rx: np.ndarray,
    h_est: np.ndarray,
    sc_indices: np.ndarray
) -> np.ndarray:
    """
    input:
        pdsch_rx : ndarray
        h_est : ndarray
        sc_indices : ndarray
    output:
        symbols : ndarray, same shape as pdsch_rx
    """
    pdsch_eq = np.zeros_like(pdsch_rx)

    threshold = 1e-10

    for i, sc in enumerate(sc_indices):
        h = h_est[sc]
        if np.abs(h) > threshold:
            pdsch_eq[i] = pdsch_rx[i] / h
        else:
            pdsch_eq[i] = pdsch_rx[i]

    return pdsch_eq


def mmse_equalize(
    pdsch_rx: np.ndarray,
    h_est: np.ndarray,
    sc_indices: np.ndarray,
    noise_var: float = 0.01
) -> np.ndarray:
    """
    input:
        pdsch_rx : ndarray
        h_est : ndarray
        sc_indices : ndarray
        noise_var : float
    output:
        symbols : ndarray, same shape as pdsch_rx
    """
    pdsch_eq = np.zeros_like(pdsch_rx)

    for i, sc in enumerate(sc_indices):
        h = h_est[sc]
        mmse_coeff = np.conj(h) / (np.abs(h) ** 2 + noise_var)
        pdsch_eq[i] = pdsch_rx[i] * mmse_coeff

    return pdsch_eq


def simulate_channel_and_equalize(
    pdsch_tx: np.ndarray,
    pdsch_sc_indices: np.ndarray,
    dmrs_rx: np.ndarray,
    dmrs_tx: np.ndarray,
    snr_db: float = 20.0,
    use_mmse: bool = True,
    noise_var: Optional[float] = None,
    n_prb: int = 20,
    dmrs_sc_per_prb: int = 6,
    n_sc: int = 240,
    dmrs_sym_positions: Tuple[int, ...] = (2, 11)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    input:
        pdsch_tx : ndarray, received data symbols
        pdsch_sc_indices : ndarray
        dmrs_rx : ndarray
        dmrs_tx : ndarray
        snr_db : float
        use_mmse : bool
        noise_var : float | none
        n_prb : int
        dmrs_sc_per_prb : int
        n_sc : int
        dmrs_sym_positions : tuple[int, ...]
    output:
        symbols : ndarray, same shape as pdsch_tx
        channel : ndarray, shape (n_sc,)
    """
    h_est = channel_estimate_full(
        dmrs_rx_all=dmrs_rx,
        dmrs_tx_all=dmrs_tx,
        dmrs_sym_positions=dmrs_sym_positions,
        n_prb=n_prb,
        dmrs_sc_per_prb=dmrs_sc_per_prb,
        n_sc=n_sc
    )

    if use_mmse:
        if noise_var is None:
            snr_linear = 10 ** (snr_db / 10.0)
            noise_var = 1.0 / snr_linear
        pdsch_eq = mmse_equalize(pdsch_tx, h_est, pdsch_sc_indices,
                                noise_var=noise_var)
    else:
        pdsch_eq = zf_equalize(pdsch_tx, h_est, pdsch_sc_indices)

    return pdsch_eq, h_est


if __name__ == "__main__":
    from dmrs import generate_dmrs_symbols

    rng = np.random.RandomState(42)
    config = ChannelConfig()
    cir = generate_tdl_channel(rng=rng)
    assert np.isclose(np.sum(np.abs(cir) ** 2), 1.0)

    delays_ns = np.array([t[0] for t in TDL_C_300NS_TAPS]) * DELAY_SCALE
    delays_samples = np.round(delays_ns / config.sample_period_ns).astype(int)
    cir_samples = np.zeros(config.fft_size, dtype=complex)
    for gain, delay in zip(cir, delays_samples):
        cir_samples[delay] += gain  # add taps that round to the same delay

    impulse = np.zeros(config.fft_size, dtype=complex)
    impulse[0] = 1.0
    np.testing.assert_allclose(
        apply_channel(impulse, cir, config), cir_samples, rtol=0, atol=1e-12
    )

    tx_dmrs = generate_dmrs_symbols(n_prb=2)
    tx_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j]) / np.sqrt(2)
    sc_indices = np.array([0, 2, 4])
    gain = 0.8 + 0.3j

    for use_mmse in (False, True):
        equalized, h_est = simulate_channel_and_equalize(
            pdsch_tx=gain * tx_symbols,
            pdsch_sc_indices=sc_indices,
            dmrs_rx=gain * tx_dmrs,
            dmrs_tx=tx_dmrs,
            use_mmse=use_mmse,
            noise_var=0.0,
            n_prb=2,
            n_sc=24
        )
        np.testing.assert_allclose(h_est, gain, rtol=0, atol=1e-12)
        np.testing.assert_allclose(equalized, tx_symbols, rtol=0, atol=1e-12)

    print("channel checks passed")
