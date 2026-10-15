#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from CRC24A import crc24a_encoder, crc24a_check
from gold_sequence import gold_sequence
from dmrs import generate_dmrs_symbols
from modulatordemodulator import ModulatorDemodulator
from re_mapping import (
    REFrameConfig,
    resource_grid_mapping,
    resource_grid_demapping,
    get_pdsch_re_positions,
    get_dmrs_re_positions,
)
from ofdm import ofdm_modulate, ofdm_demodulate
from channel_equ import (
    ChannelConfig,
    TDL_C_300NS_TAPS,
    DELAY_SCALE,
    generate_tdl_channel,
    apply_channel,
    mmse_equalize,
    zf_equalize,
    channel_estimate_full,
)
from ldpc_encoder import (
    ldpc_encode,
    select_base_graph,
    select_lifting_size,
    _BG_INFO,
)
from rate_matching import rate_match, rate_recover_llrs
from ldpc_decoder import ldpc_decode


@dataclass
class LinkConfig:

    n_prb: int = 20
    n_sc_per_prb: int = 12
    n_sym_per_slot: int = 14
    pdsch_start_sym: int = 2
    pdsch_num_sym: int = 12
    dmrs_sym_positions: Tuple[int, ...] = (2, 11)
    fft_size: int = 1024
    scs_khz: float = 30.0
    subcarrier_start: int = 0

    tb_bits: int = 76
    rv: int = 0
    qm: int = 2
    bg_id: Optional[int] = None
    target_rate: float = 0.5  # used for base graph selection

    n_rnti: int = 4660
    q_codeword: int = 0
    n_id_scramble: int = 1

    n_id_dmrs: int = 1
    n_scid: int = 0

    channel_model: str = "awgn"
    n_taps: int = 13

    csi: str = "estimated"
    eq: str = "mmse"
    ldpc_max_iter: int = 25
    ldpc_alpha: float = 0.75
    ldpc_algorithm: str = "minsum"

    @property
    def n_sc(self) -> int:
        return self.n_prb * self.n_sc_per_prb

    @property
    def cp_lengths(self) -> Tuple[int, ...]:
        return (88,) + (72,) * (self.n_sym_per_slot - 1)

    @property
    def slot_samples(self) -> int:
        return self.n_sym_per_slot * self.fft_size + int(sum(self.cp_lengths))

    def re_config(self) -> REFrameConfig:
        """
        input:
            none
        output:
            result : REFrameConfig
        """
        return REFrameConfig(
            n_prb=self.n_prb,
            n_sc_per_prb=self.n_sc_per_prb,
            n_sym_per_slot=self.n_sym_per_slot,
            dmrs_sym_positions=self.dmrs_sym_positions,
            fft_size=self.fft_size,
            scs=self.scs_khz,
            pdsch_start_sym=self.pdsch_start_sym,
            pdsch_num_sym=self.pdsch_num_sym,
        )

    def channel_config(self) -> ChannelConfig:
        """
        input:
            none
        output:
            result : ChannelConfig
        """
        return ChannelConfig(
            fft_size=self.fft_size,
            scs_khz=self.scs_khz,
            n_prb=self.n_prb,
            n_sc=self.n_sc,
        )


def link_sizes(cfg: LinkConfig) -> Dict[str, float]:
    """
    input:
        cfg : LinkConfig
    output:
        sizes : dict[str, float], link lengths and code rate
    """
    re_cfg = cfg.re_config()
    n_re_pdsch = int(get_pdsch_re_positions(re_cfg).shape[0])
    n_re_dmrs = int(get_dmrs_re_positions(re_cfg).shape[0])

    K = cfg.tb_bits + 24
    bg_id = cfg.bg_id if cfg.bg_id is not None else select_base_graph(K, cfg.target_rate)
    Z = select_lifting_size(K, bg_id)
    nb = _BG_INFO[bg_id]["nb"]
    kb = _BG_INFO[bg_id]["kb"]
    N = nb * Z
    K_sys = kb * Z
    E = n_re_pdsch * cfg.qm

    if E % cfg.qm != 0:
        raise ValueError(f"E={E} must be a multiple of Qm={cfg.qm}")
    if K > K_sys:
        raise ValueError(f"K={K} exceeds K_sys={K_sys}; increase Z or reduce tb_bits")

    return {
        "A": cfg.tb_bits,
        "K": K,
        "bg_id": bg_id,
        "Z": Z,
        "N": N,
        "K_sys": K_sys,
        "n_filler": K_sys - K,
        "E": E,
        "n_re_pdsch": n_re_pdsch,
        "n_re_dmrs": n_re_dmrs,
        "n_grid": cfg.n_sc * cfg.n_sym_per_slot,
        "n_samples": cfg.slot_samples,
        "code_rate": K / E,
    }


def pdsch_scrambling_c_init(n_rnti: int, q: int = 0, n_id: int = 1) -> int:
    """
    input:
        n_rnti : int
        q : int
        n_id : int
    output:
        c_init : int, 31-bit seed
    """
    return int(n_rnti * (2 ** 15) + q * (2 ** 14) + n_id) % (2 ** 31)


def scrambling_sequence(cfg: LinkConfig, length: int) -> np.ndarray:
    """
    input:
        cfg : LinkConfig
        length : int
    output:
        bits : ndarray, shape (length,)
    """
    c_init = pdsch_scrambling_c_init(cfg.n_rnti, cfg.q_codeword, cfg.n_id_scramble)
    return np.asarray(gold_sequence(c_init, length), dtype=np.uint8)


def scramble_bits(bits: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    input:
        bits : ndarray
        c : ndarray
    output:
        bits : ndarray, same shape as bits
    """
    return np.bitwise_xor(np.asarray(bits, dtype=np.uint8),
                          np.asarray(c, dtype=np.uint8))


def descramble_llrs(llrs: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    input:
        llrs : ndarray
        c : ndarray
    output:
        llrs : ndarray, same shape as llrs
    """
    out = np.array(llrs, dtype=np.float64, copy=True)
    out[np.asarray(c, dtype=np.uint8) == 1] *= -1.0
    return out

def cir_to_freq_response(cir: np.ndarray, cfg: LinkConfig) -> np.ndarray:
    """
    input:
        cir : ndarray
        cfg : LinkConfig
    output:
        channel : ndarray, shape (cfg.n_sc,)
    """
    ch_cfg = cfg.channel_config()
    delays_ns = np.array([t[0] for t in TDL_C_300NS_TAPS[: len(cir)]], dtype=float)
    delays_ns = delays_ns * DELAY_SCALE
    delay_samples = np.round(delays_ns / ch_cfg.sample_period_ns).astype(int)

    cir_pad = np.zeros(cfg.fft_size, dtype=complex)
    for i, d in enumerate(delay_samples):
        if d < cfg.fft_size:
            cir_pad[d] += cir[i]  # add taps that round to the same delay
    h_full = np.fft.fft(cir_pad)
    start = cfg.subcarrier_start
    return h_full[start:start + cfg.n_sc]


def apply_link_channel(
    tx_signal: np.ndarray,
    cfg: LinkConfig,
    snr_db: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    input:
        tx_signal : ndarray
        cfg : LinkConfig
        snr_db : float
        rng : RandomState
    output:
        rx_signal : ndarray, same shape as tx_signal
        info : dict, channel response and noise variances
    """
    if cfg.channel_model == "awgn":
        cir = np.array([1.0 + 0.0j])
    elif cfg.channel_model == "tdl-c":
        cir = generate_tdl_channel(n_taps=cfg.n_taps, rng=rng)
    else:
        raise ValueError(f"unknown channel model: {cfg.channel_model}")

    rx_signal = apply_channel(tx_signal, cir, cfg.channel_config())

    sigma_f2 = 10.0 ** (-snr_db / 10.0)
    sigma_t2 = cfg.fft_size * sigma_f2  # account for the receive fft scaling
    noise = np.sqrt(sigma_t2 / 2.0) * (
        rng.randn(*rx_signal.shape) + 1j * rng.randn(*rx_signal.shape)
    )
    rx_signal = rx_signal + noise

    info: Dict[str, object] = {
        "cir": cir,
        "h_freq_ideal": cir_to_freq_response(cir, cfg),
        "sigma_f2": sigma_f2,
        "sigma_t2": sigma_t2,
    }
    return rx_signal, info


@dataclass
class TxResult:
    tb_bits: np.ndarray
    data_bits: np.ndarray
    codeword: np.ndarray
    matched_bits: np.ndarray
    c_scr: np.ndarray
    scrambled_bits: np.ndarray
    symbols: np.ndarray
    dmrs_std: np.ndarray
    grid: np.ndarray
    signal: np.ndarray
    bg_id: int = 2
    Z: int = 10


def transmit(cfg: LinkConfig, tb_bits: np.ndarray,
             n_slot: int = 0) -> TxResult:
    """
    input:
        cfg : LinkConfig
        tb_bits : ndarray

        n_slot : int
    output:
        tx : TxResult, signals from each transmitter stage
    """
    tb_bits = np.asarray(tb_bits, dtype=np.uint8).ravel()
    if len(tb_bits) != cfg.tb_bits:
        raise ValueError(f"tb length {len(tb_bits)} != cfg.tb_bits={cfg.tb_bits}")

    sizes = link_sizes(cfg)
    K = int(sizes["K"])
    E = int(sizes["E"])
    N = int(sizes["N"])
    bg_id = int(sizes["bg_id"])

    data_bits = np.asarray(crc24a_encoder([int(b) for b in tb_bits]), dtype=np.uint8)
    assert len(data_bits) == K

    codeword, Z = ldpc_encode(data_bits, bg_id=bg_id, R=cfg.target_rate, return_Z=True)
    codeword = np.asarray(codeword, dtype=np.uint8)
    assert len(codeword) == N

    f = np.asarray(
        rate_match(codeword, E=E, Z=Z, K=K, bg_id=bg_id, rv=cfg.rv,
                   Qm=cfg.qm, return_interleaved=True),
        dtype=np.uint8,
    )

    c_scr = scrambling_sequence(cfg, E)
    scrambled = scramble_bits(f, c_scr)

    modulator = ModulatorDemodulator(cfg.qm)
    symbols = modulator.modulate(scrambled)

    dmrs_std = generate_dmrs_symbols(
        N_ID=cfg.n_id_dmrs, n_slot=n_slot, n_SCID=cfg.n_scid,
        dmrs_sym_positions=cfg.dmrs_sym_positions,
        n_prb=cfg.n_prb, dmrs_sc_per_prb=6,
    )
    grid = resource_grid_mapping(
        symbols,
        dmrs_std,  # pilot symbols are already in mapping order
        cfg.re_config(),
    )

    signal = ofdm_modulate(grid, cfg.fft_size, list(cfg.cp_lengths),
                           subcarrier_start=cfg.subcarrier_start)

    return TxResult(
        tb_bits=tb_bits, data_bits=data_bits, codeword=codeword, matched_bits=f,
        c_scr=c_scr, scrambled_bits=scrambled, symbols=symbols, dmrs_std=dmrs_std,
        grid=grid, signal=signal, bg_id=bg_id, Z=int(Z),
    )


@dataclass
class RxResult:
    rx_grid: np.ndarray
    pdsch_rx: np.ndarray
    dmrs_rx: np.ndarray
    h_est: np.ndarray
    pdsch_eq: np.ndarray
    llrs: np.ndarray
    cw_llrs: np.ndarray
    data_hat: np.ndarray
    tb_hat: np.ndarray
    crc_ok: bool
    ldpc_iters: int


def receive(
    cfg: LinkConfig,
    rx_signal: np.ndarray,
    tx: TxResult,
    chan_info: Dict[str, object],
) -> RxResult:
    """
    input:
        cfg : LinkConfig
        rx_signal : ndarray
        tx : TxResult, pilot and scrambling references
        chan_info : dict, ideal channel response and noise variance
    output:
        rx : RxResult, received signals and decoded bits
    """
    sizes = link_sizes(cfg)
    K = int(sizes["K"])
    E = int(sizes["E"])
    Z = int(sizes["Z"])
    bg_id = int(sizes["bg_id"])
    sigma_f2 = float(chan_info["sigma_f2"])          # type: ignore[arg-type]
    re_cfg = cfg.re_config()

    rx_grid = ofdm_demodulate(rx_signal, cfg.fft_size, list(cfg.cp_lengths),
                              cfg.n_sc, subcarrier_start=cfg.subcarrier_start)

    pdsch_rx, dmrs_rx_mapped = resource_grid_demapping(rx_grid, re_cfg)
    dmrs_rx = dmrs_rx_mapped

    if cfg.csi == "ideal":
        h_est = np.asarray(chan_info["h_freq_ideal"], dtype=np.complex128).copy()
    elif cfg.csi == "estimated":
        h_est = channel_estimate_full(
            dmrs_rx_all=dmrs_rx, dmrs_tx_all=tx.dmrs_std,
            dmrs_sym_positions=cfg.dmrs_sym_positions,
            n_prb=cfg.n_prb, dmrs_sc_per_prb=6,
            n_sc=cfg.n_sc, n_sym_slot=cfg.n_sym_per_slot,
        )
    else:
        raise ValueError(f"unknown csi mode: {cfg.csi}")

    pdsch_sc_idx = get_pdsch_re_positions(re_cfg)[:, 0]
    if cfg.eq == "mmse":
        pdsch_eq = mmse_equalize(pdsch_rx, h_est, pdsch_sc_idx, noise_var=sigma_f2)
    elif cfg.eq == "zf":
        pdsch_eq = zf_equalize(pdsch_rx, h_est, pdsch_sc_idx)
    else:
        raise ValueError(f"unknown equalizer: {cfg.eq}")

    modulator = ModulatorDemodulator(cfg.qm)
    h_per_re = h_est[pdsch_sc_idx]
    llrs_rx = modulator.demodulate(pdsch_rx, noise_var=sigma_f2, channel=h_per_re)  # use received symbols and per-resource channel gains

    llrs = descramble_llrs(llrs_rx, tx.c_scr)  # flip the soft-bit signs for scrambled positions

    cw_llrs = rate_recover_llrs(llrs, E=E, Z=Z, K=K, bg_id=bg_id,
                                rv=cfg.rv, Qm=cfg.qm)

    data_hat, iters = ldpc_decode(
        cw_llrs, Z=Z, bg_id=bg_id, K=K,
        max_iter=cfg.ldpc_max_iter, alpha=cfg.ldpc_alpha,
        algorithm=cfg.ldpc_algorithm, return_info=True, return_iter=True,
    )
    data_hat = np.asarray(data_hat, dtype=np.uint8)

    tb_list, crc_ok = crc24a_check([int(b) for b in data_hat])
    tb_hat = np.asarray(tb_list, dtype=np.uint8)

    return RxResult(
        rx_grid=rx_grid, pdsch_rx=pdsch_rx, dmrs_rx=dmrs_rx, h_est=h_est,
        pdsch_eq=pdsch_eq, llrs=llrs, cw_llrs=cw_llrs, data_hat=data_hat,
        tb_hat=tb_hat, crc_ok=bool(crc_ok), ldpc_iters=int(iters),
    )


@dataclass
class LinkResult:
    snr_db: float
    sizes: Dict[str, float]
    tb_bits: np.ndarray
    tb_hat: np.ndarray
    crc_ok: bool
    tb_err_bits: int
    blk_err: bool
    evm: float
    h_mse: float
    ldpc_iters: int
    times: Dict[str, float] = field(default_factory=dict)
    tx: Optional[TxResult] = None
    rx: Optional[RxResult] = None
    chan_info: Optional[Dict[str, object]] = None

    @property
    def ber(self) -> float:
        return self.tb_err_bits / len(self.tb_bits)

    def summary(self) -> Dict[str, float]:
        """
        input:
            none
        output:
            result : dict[str, float]
        """
        return {
            "snr_db": self.snr_db,
            "blk_err": 1.0 if self.blk_err else 0.0,
            "tb_err_bits": float(self.tb_err_bits),
            "crc_ok": 1.0 if self.crc_ok else 0.0,
            "evm": self.evm,
            "h_mse": self.h_mse,
            "ldpc_iters": float(self.ldpc_iters),
        }


def simulate_slot(
    cfg: Optional[LinkConfig] = None,
    snr_db: float = 10.0,
    tb_bits: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
    n_slot: int = 0,
    keep_arrays: bool = True,
) -> LinkResult:
    """
    input:
        cfg : LinkConfig | none
        snr_db : float
        tb_bits : ndarray | none
        rng : RandomState | none
        n_slot : int
        keep_arrays : bool
    output:
        result : LinkResult, decoded block and link metrics
    """
    if cfg is None:
        cfg = LinkConfig()
    if rng is None:
        rng = np.random.RandomState(0)
    if tb_bits is None:
        tb_bits = rng.randint(0, 2, size=cfg.tb_bits).astype(np.uint8)

    sizes = link_sizes(cfg)

    t0 = time.perf_counter()
    tx = transmit(cfg, tb_bits, n_slot=n_slot)
    t1 = time.perf_counter()

    rx_signal, chan_info = apply_link_channel(tx.signal, cfg, snr_db, rng)
    t2 = time.perf_counter()

    rx = receive(cfg, rx_signal, tx, chan_info)
    t3 = time.perf_counter()

    tb_err_bits = int(np.sum(rx.tb_hat != tx.tb_bits))
    crc_ok = rx.crc_ok
    blk_err = (not crc_ok) or (tb_err_bits > 0)
    evm = float(np.sqrt(np.mean(np.abs(rx.pdsch_eq - tx.symbols) ** 2)))
    h_ideal = np.asarray(chan_info["h_freq_ideal"], dtype=np.complex128)
    h_mse = float(np.mean(np.abs(rx.h_est - h_ideal) ** 2))

    return LinkResult(
        snr_db=snr_db, sizes=sizes, tb_bits=tx.tb_bits, tb_hat=rx.tb_hat,
        crc_ok=crc_ok, tb_err_bits=tb_err_bits, blk_err=blk_err, evm=evm,
        h_mse=h_mse, ldpc_iters=rx.ldpc_iters,
        times={"tx": t1 - t0, "channel": t2 - t1, "rx": t3 - t2, "total": t3 - t0},
        tx=tx if keep_arrays else None,
        rx=rx if keep_arrays else None,
        chan_info=chan_info if keep_arrays else None,
    )


def bler_sweep(
    snr_db_list: List[float],
    n_slots: int = 50,
    cfg: Optional[LinkConfig] = None,
    seed: int = 0,
    verbose: bool = False,
) -> List[Dict[str, float]]:
    """
    input:
        snr_db_list : list[float]
        n_slots : int
        cfg : LinkConfig | none
        seed : int
        verbose : bool
    output:
        results : list[dict[str, float]], one entry per snr point
    """
    if cfg is None:
        cfg = LinkConfig()

    results: List[Dict[str, float]] = []
    for idx, snr in enumerate(snr_db_list):
        rng = np.random.RandomState(seed + 1000 * idx)
        n_err = 0
        n_bit_err = 0
        n_bit_total = 0
        evm_sum = 0.0
        for _ in range(n_slots):
            res = simulate_slot(cfg=cfg, snr_db=float(snr), rng=rng,
                                n_slot=int(rng.randint(0, 20)), keep_arrays=False)
            n_err += 1 if res.blk_err else 0
            n_bit_err += res.tb_err_bits
            n_bit_total += len(res.tb_bits)
            evm_sum += res.evm
        bler = n_err / n_slots
        out = {
            "snr_db": float(snr),
            "n_slots": float(n_slots),
            "n_err": float(n_err),
            "bler": float(bler),
            "ber": float(n_bit_err / max(n_bit_total, 1)),
            "evm": float(evm_sum / n_slots),
        }
        results.append(out)
        if verbose:
            print(f"  SNR = {snr:6.1f} dB | BLER = {bler:6.3f} ({n_err:3d}/{n_slots})"
                  f" | BER = {out['ber']:.2e} | EVM = {out['evm']:.3f}")
    return results


def measure_noise_scaling(cfg: LinkConfig, snr_db: float,
                          rng: np.random.RandomState) -> Tuple[float, float]:
    """
    input:
        cfg : LinkConfig
        snr_db : float
        rng : RandomState
    output:
        measured, expected : float, frequency-domain noise variance
    """
    sigma_f2 = 10.0 ** (-snr_db / 10.0)
    sigma_t2 = cfg.fft_size * sigma_f2  # account for the receive fft scaling
    noise = np.sqrt(sigma_t2 / 2.0) * (
        rng.randn(cfg.slot_samples) + 1j * rng.randn(cfg.slot_samples)
    )
    grid = ofdm_demodulate(noise, cfg.fft_size, list(cfg.cp_lengths), cfg.n_sc,
                           subcarrier_start=cfg.subcarrier_start)
    return float(np.mean(np.abs(grid) ** 2)), float(sigma_f2)


if __name__ == "__main__":
    cfg = LinkConfig()
    measured, expected = measure_noise_scaling(
        cfg, snr_db=10.0, rng=np.random.RandomState(2024)
    )
    assert abs(measured - expected) / expected < 0.10

    cases = [
        ("awgn", "ideal", 80.0, 7, 0),
        ("awgn", "estimated", 20.0, 11, 0),
        ("tdl-c", "estimated", 20.0, 99, 0),
        ("tdl-c", "ideal", 80.0, 99, 392),
    ]
    for model, csi, snr_db, seed, start in cases:
        cfg = LinkConfig(
            channel_model=model, csi=csi, subcarrier_start=start
        )
        result = simulate_slot(
            cfg=cfg, snr_db=snr_db, rng=np.random.RandomState(seed)
        )
        assert result.crc_ok and result.tb_err_bits == 0
        if csi == "ideal":
            assert result.h_mse < 1e-12
            assert result.evm < 1e-3

    print("pdsch e2e checks passed")
