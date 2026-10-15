#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta, norm

from pdsch_e2e import LinkConfig, link_sizes, simulate_slot, apply_link_channel


DATA_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"


def slot_duration_ms(cfg: LinkConfig) -> float:
    """
    input:
        cfg : LinkConfig
    output:
        duration : float, milliseconds
    """
    return 15.0 / float(cfg.scs_khz)


def ebn0_db_from_snr(snr_db: float, n_re: int, k: int) -> float:
    """
    input:
        snr_db : float
        n_re : int
        k : int
    output:
        ebn0_db : float, based on k bits
    """
    return float(snr_db + 10.0 * np.log10(n_re / k))


def goodput_mbps(k: int, bler: float, slot_ms: float) -> float:
    """
    input:
        k : int
        bler : float
        slot_ms : float
    output:
        goodput : float, megabits per second based on k bits
    """
    return float(k * (1.0 - bler) / (slot_ms * 1e-3) / 1e6)


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    input:
        k : int
        n : int
        z : float
    output:
        lower, upper : float
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """
    input:
        k : int
        n : int
        conf : float
    output:
        lower, upper : float
    """
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def rule_of_three_upper(n: int) -> float:
    """
    input:
        n : int
    output:
        upper : float, approximate error probability bound
    """
    return 0.0 if n <= 0 else float(min(1.0, 3.0 / n))


def qpsk_ber_theory(ebn0_db: float) -> float:
    """
    input:
        ebn0_db : float
    output:
        ber : float
    """
    return float(norm.sf(np.sqrt(2.0 * 10 ** (ebn0_db / 10.0))))


def shannon_limit_complex_db(rate: float) -> float:
    """
    input:
        rate : float
    output:
        ebn0_db : float
    """
    if rate <= 0:
        return float(-np.inf)
    return float(10.0 * np.log10((2.0 ** rate - 1.0) / rate))


_U_GRID = np.linspace(-12.0, 12.0, 6001)
_U_WEIGHT = norm.pdf(_U_GRID)


def biawgn_capacity(esn0_lin: float) -> float:
    """
    input:
        esn0_lin : float
    output:
        capacity : float, bits per real channel use
    """
    arg = 4.0 * esn0_lin + np.sqrt(8.0 * esn0_lin) * _U_GRID
    return float(1.0 - np.trapezoid(np.log2(1.0 + np.exp(-arg)) * _U_WEIGHT, _U_GRID))


def shannon_limit_biawgn_db(rate_bits_per_use: float) -> float:
    """
    input:
        rate_bits_per_use : float
    output:
        ebn0_db : float
    """
    from scipy.optimize import brentq

    rate = float(rate_bits_per_use)
    if rate <= 0:
        return float(-np.inf)
    if rate >= 0.999:
        return float("inf")
    f = lambda db: biawgn_capacity(rate * 10 ** (db / 10.0)) - rate
    try:
        return float(brentq(f, -5.0, 30.0, xtol=1e-4))
    except ValueError:
        return float("nan")


@dataclass
class PointResult:
    snr_db: float
    n_blocks: int
    n_errors: int
    bler: float
    bler_lo: float
    bler_hi: float
    ber: float
    n_bit_errors: int
    n_bits: int
    evm_mean: float
    h_mse_mean: float
    ldpc_iters_mean: float
    eb_n0_db: float
    goodput_mbps: float
    elapsed_s: float
    stop_reason: str

    @property
    def bler_upper_if_zero(self) -> Optional[float]:
        return rule_of_three_upper(self.n_blocks) if self.n_errors == 0 else None



@dataclass
class CurveResult:
    label: str
    cfg: LinkConfig
    points: List[PointResult] = field(default_factory=list)
    sizes: Dict[str, float] = field(default_factory=dict)

    @property
    def snr_db(self) -> np.ndarray:
        return np.array([p.snr_db for p in self.points], dtype=float)

    @property
    def bler(self) -> np.ndarray:
        return np.array([p.bler for p in self.points], dtype=float)

    @property
    def bler_lo(self) -> np.ndarray:
        return np.array([p.bler_lo for p in self.points], dtype=float)

    @property
    def bler_hi(self) -> np.ndarray:
        return np.array([p.bler_hi for p in self.points], dtype=float)

    @property
    def ber(self) -> np.ndarray:
        return np.array([p.ber for p in self.points], dtype=float)

    @property
    def evm(self) -> np.ndarray:
        return np.array([p.evm_mean for p in self.points], dtype=float)

    @property
    def h_mse(self) -> np.ndarray:
        return np.array([p.h_mse_mean for p in self.points], dtype=float)

    @property
    def ebn0_db(self) -> np.ndarray:
        return np.array([p.eb_n0_db for p in self.points], dtype=float)

    @property
    def goodput(self) -> np.ndarray:
        return np.array([p.goodput_mbps for p in self.points], dtype=float)

    def waterfall_snr(self, target_bler: float = 0.1) -> Optional[float]:
        """
        input:
            target_bler : float
        output:
            snr_db : float | none, estimated target crossing
        """
        snr, bler = self.snr_db, self.bler
        if len(snr) < 2:
            return None
        eff = np.array([max(p.bler, 0.5 / max(p.n_blocks, 1)) for p in self.points])  # keep zero-error points finite for interpolation
        y = np.log10(np.clip(eff, 1e-12, 1.0))
        t = np.log10(target_bler)
        order = np.argsort(snr)
        snr, y = snr[order], y[order]
        y = np.minimum.accumulate(y)  # use a monotone envelope for the crossing estimate
        if not (y[0] >= t >= y[-1]):
            return None
        idx = int(np.where(y <= t)[0][0])
        if idx == 0:
            return float(snr[0])
        x0, x1 = snr[idx - 1], snr[idx]
        y0, y1 = y[idx - 1], y[idx]
        if y1 == y0:
            return float(x1)
        return float(x0 + (t - y0) * (x1 - x0) / (y1 - y0))

    def waterfall_ebn0(self, target_bler: float = 0.1) -> Optional[float]:
        """
        input:
            target_bler : float
        output:
            ebn0_db : float | none, estimated target crossing
        """
        snr_hit = self.waterfall_snr(target_bler)
        if snr_hit is None:
            return None
        n_re = int(self.sizes.get("n_re_pdsch", 0)) or 1
        k = int(self.sizes.get("K", 0)) or 1
        return ebn0_db_from_snr(snr_hit, n_re, k)

    def max_goodput(self) -> Tuple[float, float]:
        """
        input:
            none
        output:
            goodput_mbps, snr_db : float
        """
        if not self.points:
            return (0.0, float("nan"))
        i = int(np.argmax(self.goodput))
        return (float(self.goodput[i]), float(self.snr_db[i]))

    def theory_gap_db(self, target_bler: float = 0.1) -> Optional[float]:
        """
        input:
            target_bler : float
        output:
            gap_db : float | none
        """
        ebn0 = self.waterfall_ebn0(target_bler)
        if ebn0 is None:
            return None
        n_re = int(self.sizes.get("n_re_pdsch", 0)) or 1
        k = int(self.sizes.get("K", 0)) or 1
        rate_per_real = k / (2.0 * n_re)  # each qpsk symbol has two real dimensions
        limit = shannon_limit_biawgn_db(rate_per_real)
        return float(ebn0 - limit) if np.isfinite(limit) else None

    def description(self) -> str:
        """
        input:
            none
        output:
            description : str
        """
        c = self.cfg
        s = self.sizes
        return (f"{self.label}: PRB={c.n_prb}, TB={c.tb_bits}b (K={int(s.get('K', 0))}), "
                f"E={int(s.get('E', 0))}, N={int(s.get('N', 0))}, "
                f"rate={s.get('code_rate', 0):.3f}, {c.channel_model.upper()}, "
                f"CSI={c.csi}, eq={c.eq.upper()}, Qm={c.qm}")


def simulate_point(
    cfg: LinkConfig,
    snr_db: float,
    max_blocks: int = 300,
    min_blocks: int = 20,
    target_errors: int = 30,
    seed: int = 0,
    verbose: bool = False,
) -> PointResult:
    """
    input:
        cfg : LinkConfig
        snr_db : float
        max_blocks : int
        min_blocks : int
        target_errors : int
        seed : int
        verbose : bool
    output:
        result : PointResult, statistics for one snr point
    """
    rng = np.random.RandomState(seed)
    sizes = link_sizes(cfg)
    k = int(sizes["K"])
    n_re = int(sizes["n_re_pdsch"])
    slot_ms = slot_duration_ms(cfg)

    n_blocks = 0
    n_errors = 0
    n_bit_errors = 0
    n_bits = 0
    evm_list: List[float] = []
    mse_list: List[float] = []
    iters_list: List[float] = []
    reason = "max_blocks"

    t0 = time.perf_counter()
    while True:
        res = simulate_slot(
            cfg=cfg, snr_db=float(snr_db), rng=rng,
            n_slot=int(rng.randint(0, 20)),
            keep_arrays=False,
        )
        n_blocks += 1
        if res.blk_err:
            n_errors += 1
        n_bit_errors += res.tb_err_bits
        n_bits += len(res.tb_bits)
        evm_list.append(res.evm)
        mse_list.append(res.h_mse)
        iters_list.append(float(res.ldpc_iters))

        if n_blocks >= max_blocks:
            reason = "max_blocks"
            break
        if n_blocks >= min_blocks and n_errors >= target_errors:  # stop once enough error blocks have been observed
            reason = "target_errors"
            break
    elapsed = time.perf_counter() - t0

    bler = n_errors / n_blocks
    lo, hi = wilson_interval(n_errors, n_blocks)
    point = PointResult(
        snr_db=float(snr_db),
        n_blocks=n_blocks,
        n_errors=n_errors,
        bler=float(bler),
        bler_lo=lo,
        bler_hi=hi,
        ber=float(n_bit_errors / max(n_bits, 1)),
        n_bit_errors=int(n_bit_errors),
        n_bits=int(n_bits),
        evm_mean=float(np.mean(evm_list)),
        h_mse_mean=float(np.mean(mse_list)),
        ldpc_iters_mean=float(np.mean(iters_list)),
        eb_n0_db=ebn0_db_from_snr(snr_db, n_re, k),
        goodput_mbps=goodput_mbps(cfg.tb_bits, bler, slot_ms),
        elapsed_s=elapsed,
        stop_reason=reason,
    )
    if verbose:
        extra = (f", upper bound {point.bler_upper_if_zero:.2e}"
                 if point.bler_upper_if_zero is not None else "")
        print(f"    SNR {snr_db:+6.1f} dB: BLER = {bler:6.3f} "
              f"({n_errors:3d}/{n_blocks:3d}){extra}, BER = {point.ber:.2e}, "
              f"goodput = {point.goodput_mbps:6.3f} Mbps, {elapsed:5.1f}s")
    return point


def sweep(
    cfg: LinkConfig,
    snr_db_list: Sequence[float],
    label: Optional[str] = None,
    max_blocks: int = 300,
    min_blocks: int = 20,
    target_errors: int = 30,
    seed: int = 0,
    verbose: bool = True,
) -> CurveResult:
    """
    input:
        cfg : LinkConfig
        snr_db_list : sequence[float]
        label : str | none
        max_blocks : int
        min_blocks : int
        target_errors : int
        seed : int
        verbose : bool
    output:
        curve : CurveResult
    """
    sizes = link_sizes(cfg)
    if label is None:
        label = (f"PRB{cfg.n_prb}_TB{cfg.tb_bits}_{2 ** cfg.qm}QAM_"
                 f"{cfg.channel_model}_{cfg.csi}")
    curve = CurveResult(label=label, cfg=cfg, sizes=sizes)
    if verbose:
        print(f"  [{label}] {curve.description()}")
    for i, snr in enumerate(snr_db_list):
        curve.points.append(
            simulate_point(cfg, float(snr), max_blocks=max_blocks,
                           min_blocks=min_blocks, target_errors=target_errors,
                           seed=seed + 1000 * i, verbose=verbose)
        )
    return curve


def link_adaptation(
    curves: Sequence[CurveResult],
    snr_grid: Optional[Sequence[float]] = None,
    target_bler: Optional[float] = 0.1,
) -> List[Dict[str, float]]:
    """
    input:
        curves : sequence[CurveResult]
        snr_grid : sequence[float] | none
        target_bler : float | none
    output:
        choices : list[dict], selected curve at each matching snr
    """
    if snr_grid is None:
        grid = sorted({float(s) for c in curves for s in c.snr_db})
    else:
        grid = [float(s) for s in snr_grid]

    out: List[Dict[str, float]] = []
    for snr in grid:
        best = None
        for c in curves:
            if len(c.points) == 0:
                continue
            i = int(np.argmin(np.abs(c.snr_db - snr)))
            p = c.points[i]
            if abs(p.snr_db - snr) > 1e-9:
                continue
            if target_bler is not None and p.bler_hi > target_bler:  # use the confidence bound for the reliability constraint
                continue
            if best is None or p.goodput_mbps > best[0]:
                best = (p.goodput_mbps, c.label, p, c)
        if best is not None:
            out.append({
                "snr_db": float(snr),
                "label": best[1],
                "goodput_mbps": float(best[0]),
                "bler": float(best[2].bler),
                "rate": float(best[3].sizes.get("code_rate", 0.0)),
            })
    return out


def save_curve_csv(curve: CurveResult, path: Optional[Path] = None) -> Path:
    """
    input:
        curve : CurveResult
        path : Path | none
    output:
        path : Path, saved csv file
    """
    path = Path(path) if path else (DATA_DIR / f"bler_curve_{curve.label}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write(f"# {curve.description()}\n")
        meta = {"label": curve.label, "sizes": curve.sizes, "cfg": asdict(curve.cfg)}
        f.write("# meta: " + json.dumps(meta, ensure_ascii=False) + "\n")
        w = csv.writer(f)
        w.writerow(["snr_db", "ebn0_db", "n_blocks", "n_errors", "bler",
                    "bler_ci_low", "bler_ci_high", "bler_upper_if_zero",
                    "ber", "n_bit_errors", "n_bits", "goodput_mbps",
                    "evm_mean", "h_mse_mean", "ldpc_iters_mean",
                    "elapsed_s", "stop_reason"])
        for p in curve.points:
            up = p.bler_upper_if_zero
            w.writerow([f"{p.snr_db:.2f}", f"{p.eb_n0_db:.3f}", p.n_blocks, p.n_errors,
                        f"{p.bler:.6f}", f"{p.bler_lo:.6f}", f"{p.bler_hi:.6f}",
                        "" if up is None else f"{up:.6f}",
                        f"{p.ber:.6e}", p.n_bit_errors, p.n_bits,
                        f"{p.goodput_mbps:.4f}", f"{p.evm_mean:.5f}",
                        f"{p.h_mse_mean:.6f}", f"{p.ldpc_iters_mean:.2f}",
                        f"{p.elapsed_s:.2f}", p.stop_reason])
    return path


def load_curve_csv(path: Path) -> CurveResult:
    """
    input:
        path : Path
    output:
        curve : CurveResult
    """
    path = Path(path)
    meta: Optional[dict] = None
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("# meta:"):
                meta = json.loads(line.split("meta:", 1)[1].strip())
                continue
            if line.startswith("#"):
                continue
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) < 17 or parts[0] in ("", "snr_db"):
                continue
            rows.append(parts)
    if meta is None:
        raise ValueError(f"{path.name} is missing '# meta:' metadata")

    curve = CurveResult(label=meta.get("label", path.stem),
                        cfg=LinkConfig(**meta["cfg"]),
                        sizes=meta["sizes"])
    for p in rows:
        curve.points.append(PointResult(
            snr_db=float(p[0]), n_blocks=int(p[2]), n_errors=int(p[3]),
            bler=float(p[4]), bler_lo=float(p[5]), bler_hi=float(p[6]),
            ber=float(p[8]), n_bit_errors=int(p[9]), n_bits=int(p[10]),
            goodput_mbps=goodput_mbps(
                curve.cfg.tb_bits, float(p[4]), slot_duration_ms(curve.cfg)
            ),
            evm_mean=float(p[12]),
            h_mse_mean=float(p[13]), ldpc_iters_mean=float(p[14]),
            eb_n0_db=float(p[1]), elapsed_s=float(p[15]), stop_reason=p[16],
        ))
    return curve


def print_summary(curves: Sequence[CurveResult],
                  targets: Sequence[float] = (0.1, 0.01)) -> None:
    """
    input:
        curves : sequence[CurveResult]
        targets : sequence[float]
    output:
        none
    """
    print("curve summary")
    for c in curves:
        rate = c.sizes.get("code_rate", 0.0)
        k = int(c.sizes.get("K", 0))
        n_re = int(c.sizes.get("n_re_pdsch", 0))
        slot_ms = slot_duration_ms(c.cfg)
        print(f"[{c.label}] rate={rate:.4f}, K={k}, N_RE={n_re}")
        for t in targets:
            snr_hit = c.waterfall_snr(t)
            if snr_hit is None:
                print(f"  BLER={t:g}: target not crossed in scanned range")
                continue
            ebn0_hit = ebn0_db_from_snr(snr_hit, n_re, k)
            line = f"  BLER={t:g}: SNR={snr_hit:+.2f} dB, Eb/N0={ebn0_hit:+.2f} dB"
            if abs(t - targets[0]) < 1e-12:
                gap = c.theory_gap_db(t)
                if gap is not None:
                    line += f", BIAWGN reference gap={gap:.2f} dB"
            print(line)
        gp, gp_snr = c.max_goodput()
        cap = goodput_mbps(c.cfg.tb_bits, 0.0, slot_ms)
        print(f"  peak goodput={gp:.3f} Mbps at SNR={gp_snr:+.2f} dB; payload limit={cap:.3f} Mbps")


def _setup_matplotlib():
    """
    input:
        none
    output:
        plotting : matplotlib.pyplot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    return plt


def plot_curves(curves: Sequence[CurveResult],
                outdir: Optional[Path] = None) -> List[Path]:
    """
    input:
        curves : sequence[CurveResult]
        outdir : Path | none
    output:
        paths : list[Path], saved figure files
    """
    plt = _setup_matplotlib()
    outdir = Path(outdir) if outdir else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []

    def _bler_panel(ax):
        for c in curves:
            yerr = np.vstack([np.maximum(c.bler - c.bler_lo, 0.0),
                              np.maximum(c.bler_hi - c.bler, 0.0)])
            ax.errorbar(c.snr_db, np.where(c.bler > 0, c.bler, np.nan), yerr=yerr,
                        marker="o", capsize=3, lw=1.5, label=c.label)
            for p in c.points:
                if p.n_errors == 0:  # show an upper bound for zero observed errors
                    ax.plot(p.snr_db, rule_of_three_upper(p.n_blocks), "v",
                            mfc="none", ms=7)
        ax.axhline(0.1, color="gray", ls=":", lw=1, label="BLER = 0.1")
        ax.plot([], [], "v", mfc="none", ms=7, color="gray",
                label="zero block errors: approx. 95% upper bound 3/N")
        ax.set_xlabel("SNR (dB per RE)")
        ax.set_ylabel("BLER")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower left")
        ax.set_title("BLER vs SNR")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _bler_panel(axes[0])
    for c in curves:
        axes[1].plot(c.snr_db, c.goodput, marker="s", lw=1.5, label=c.label)
    axes[1].set_xlabel("SNR (dB per RE)")
    axes[1].set_ylabel("payload goodput (Mbps)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    axes[1].set_title("Goodput vs SNR")
    fig.tight_layout()
    p1 = outdir / "bler_throughput.png"
    fig.savefig(p1)
    plt.close(fig)
    files.append(p1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in curves:
        y = np.array([p.ber if p.n_bit_errors > 0 else np.nan for p in c.points])
        axes[0].semilogy(c.snr_db, y, marker="^", lw=1.5, label=c.label)
    axes[0].set_xlabel("SNR (dB per RE)")
    axes[0].set_ylabel("payload BER (zero-error points omitted)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("BER vs SNR")
    for c in curves:
        axes[1].plot(c.snr_db, c.evm, marker="d", lw=1.5, label=c.label)
    axes[1].set_xlabel("SNR (dB per RE)")
    axes[1].set_ylabel("post-equalization EVM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    axes[1].set_title("EVM vs SNR")
    fig.tight_layout()
    p2 = outdir / "ber_evm.png"
    fig.savefig(p2)
    plt.close(fig)
    files.append(p2)
    return files


def plot_ebn0_shannon(curves: Sequence[CurveResult],
                      outdir: Optional[Path] = None) -> Path:
    """
    input:
        curves : sequence[CurveResult]
        outdir : Path | none
    output:
        path : Path, saved figure file
    """
    plt = _setup_matplotlib()
    outdir = Path(outdir) if outdir else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    limits: List[float] = []
    for c in curves:
        ax.semilogy(c.ebn0_db, np.where(c.bler > 0, c.bler, np.nan), marker="o", lw=1.5,
                    label=f"{c.label} (rate={c.sizes.get('code_rate', 0):.3f})")
        n_re = int(c.sizes.get("n_re_pdsch", 1))
        k = int(c.sizes.get("K", 1))
        limit = shannon_limit_biawgn_db(k / (2.0 * n_re))
        if np.isfinite(limit):
            limits.append(limit)
        ebn0 = np.linspace(max(0.0, c.ebn0_db.min()), c.ebn0_db.max() + 2, 50)
        if c is curves[0]:
            ax.semilogy(ebn0, [qpsk_ber_theory(x) for x in ebn0], "k:",
                        lw=1, label="uncoded QPSK theoretical BER")
    if limits:
        ax.axvspan(min(limits), max(limits), color="tab:red", alpha=0.12)
        ax.text(min(limits), 3e-5,
                f"BIAWGN reference {min(limits):.2f} ~ {max(limits):.2f} dB",
                fontsize=7, color="tab:red")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BLER / BER")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("BLER vs Eb/N0 and BIAWGN reference")
    fig.tight_layout()
    path = outdir / "bler_ebn0_shannon.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def simulate_uncoded_point(
    cfg: LinkConfig,
    snr_db: float,
    n_slots: int = 20,
    seed: int = 0,
    ideal_csi: bool = False,
) -> Tuple[float, float, int]:
    """
    input:
        cfg : LinkConfig
        snr_db : float
        n_slots : int
        seed : int
        ideal_csi : bool
    output:
        ber, ebn0_db : float
        n_bits : int
    """
    from dmrs import generate_dmrs_symbols
    from modulatordemodulator import ModulatorDemodulator
    from ofdm import ofdm_modulate, ofdm_demodulate
    from re_mapping import (resource_grid_mapping, resource_grid_demapping,
                            get_pdsch_re_positions)
    from channel_equ import channel_estimate_full

    rng = np.random.RandomState(seed)
    re_cfg = cfg.re_config()
    sizes = link_sizes(cfg)
    n_re = int(sizes["n_re_pdsch"])
    cp = list(cfg.cp_lengths)
    modulator = ModulatorDemodulator(cfg.qm)
    pdsch_sc_idx = get_pdsch_re_positions(re_cfg)[:, 0]

    n_bit_err = 0
    n_bits = 0
    for _ in range(n_slots):
        bits = rng.randint(0, 2, size=n_re * cfg.qm).astype(np.uint8)
        symbols = modulator.modulate(bits)
        dmrs = generate_dmrs_symbols(
            N_ID=cfg.n_id_dmrs, n_slot=int(rng.randint(0, 20)), n_SCID=cfg.n_scid,
            dmrs_sym_positions=cfg.dmrs_sym_positions,
            n_prb=cfg.n_prb, dmrs_sc_per_prb=6,
        )
        grid = resource_grid_mapping(symbols, dmrs, re_cfg)
        tx = ofdm_modulate(grid, cfg.fft_size, cp,
                           subcarrier_start=cfg.subcarrier_start)
        rx, info = apply_link_channel(tx, cfg, snr_db, rng)
        rx_grid = ofdm_demodulate(rx, cfg.fft_size, cp, cfg.n_sc,
                                  subcarrier_start=cfg.subcarrier_start)
        pdsch_rx, dmrs_rx = resource_grid_demapping(rx_grid, re_cfg)
        if ideal_csi:
            h_est = np.asarray(info["h_freq_ideal"], dtype=np.complex128)
        else:
            h_est = channel_estimate_full(
                dmrs_rx_all=dmrs_rx, dmrs_tx_all=dmrs,
                dmrs_sym_positions=cfg.dmrs_sym_positions,
                n_prb=cfg.n_prb, dmrs_sc_per_prb=6,
                n_sc=cfg.n_sc, n_sym_slot=cfg.n_sym_per_slot,
            )
        llrs = modulator.demodulate(pdsch_rx, noise_var=float(info["sigma_f2"]),
                                    channel=h_est[pdsch_sc_idx])
        hard = (llrs < 0).astype(np.uint8)
        n_bit_err += int(np.sum(hard != bits))
        n_bits += int(bits.size)

    ber = n_bit_err / max(n_bits, 1)
    ebn0 = ebn0_db_from_snr(snr_db, n_re, n_re * cfg.qm)
    return ber, ebn0, n_bits


def phy_validation(n_slots: int = 20, verbose: bool = True) -> List[Dict[str, float]]:
    """
    input:
        n_slots : int
        verbose : bool
    output:
        results : list[dict[str, float]], measured and theoretical ber
    """
    cfg = LinkConfig(n_prb=20, tb_bits=76, channel_model="awgn")
    snr_list = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    out: List[Dict[str, float]] = []
    if verbose:
        print("uncoded QPSK validation against AWGN theory")
        print(f"{'SNR':>6s} {'Eb/N0':>7s} | {'ideal BER':>12s} {'estimated BER':>12s} "
              f"{'theory BER':>11s} {'ideal loss':>9s} {'estimated loss':>9s}")
    for i, snr in enumerate(snr_list):
        ber_i, ebn0, nbits = simulate_uncoded_point(cfg, snr, n_slots=n_slots,
                                                    seed=7000 + 100 * i, ideal_csi=True)
        ber_e, _, _ = simulate_uncoded_point(cfg, snr, n_slots=n_slots,
                                             seed=7000 + 100 * i, ideal_csi=False)
        th = qpsk_ber_theory(ebn0)
        loss_i = _loss_db(ber_i, ebn0)
        loss_e = _loss_db(ber_e, ebn0)
        row = {"snr_db": float(snr), "ebn0_db": float(ebn0), "ber_ideal": float(ber_i),
               "ber_estimated": float(ber_e), "ber_theory": float(th),
               "loss_ideal_db": loss_i, "loss_estimated_db": loss_e,
               "n_bits": float(nbits)}
        out.append(row)
        if verbose:
            print(f"  {snr:+6.1f} {ebn0:+7.2f} | {ber_i:12.3e} {ber_e:12.3e} {th:11.3e} "
                  f"{_fmt(loss_i):>9s} {_fmt(loss_e):>9s}")
    return out


def _loss_db(ber_measured: float, ebn0_db: float) -> float:
    """
    input:
        ber_measured : float
        ebn0_db : float
    output:
        loss_db : float
    """
    ber = float(ber_measured)
    if not (0.0 < ber <= 0.15):
        return float("nan")
    gamma_req = (norm.isf(ber) ** 2) / 2.0
    gamma_act = 10 ** (ebn0_db / 10.0)
    return float(10.0 * np.log10(gamma_act / gamma_req))


def _fmt(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:+.2f} dB"


def plot_phy_validation(rows: Sequence[Dict[str, float]],
                        outdir: Optional[Path] = None) -> Path:
    """
    input:
        rows : sequence[dict[str, float]]
        outdir : Path | none
    output:
        path : Path, saved figure file
    """
    plt = _setup_matplotlib()
    outdir = Path(outdir) if outdir else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ebn0 = np.array([r["ebn0_db"] for r in rows])
    ax.semilogy(ebn0, [r["ber_ideal"] for r in rows], "o-", label="measured (ideal CSI)")
    ax.semilogy(ebn0, [r["ber_estimated"] for r in rows], "s-", label="measured (DM-RS estimated CSI)")
    grid = np.linspace(ebn0.min() - 1, ebn0.max() + 1, 60)
    ax.semilogy(grid, [qpsk_ber_theory(x) for x in grid], "k--",
                label="theory: Q(sqrt(2*Eb/N0))")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("uncoded BER")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("uncoded QPSK: measured vs theory (AWGN)")
    fig.tight_layout()
    path = outdir / "phy_validation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def preset_curves() -> List[Tuple[str, LinkConfig, List[float]]]:
    return [
        ("20PRB_r0.019_awgn",
         LinkConfig(n_prb=20, tb_bits=76, channel_model="awgn", csi="estimated"),
         [-9.0, -8.5, -8.0, -7.5, -7.0, -6.5]),
        ("20PRB_r0.019_tdlc",
         LinkConfig(n_prb=20, tb_bits=76, channel_model="tdl-c", csi="estimated"),
         [-7.0, -6.5, -6.0, -5.5, -5.0, -4.5]),
        ("2PRB_r0.19_awgn",
         LinkConfig(n_prb=2, tb_bits=76, channel_model="awgn", csi="estimated"),
         [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]),
        ("2PRB_r0.19_tdlc",
         LinkConfig(n_prb=2, tb_bits=76, channel_model="tdl-c", csi="estimated"),
         [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]),
        ("2PRB_r0.33_awgn",
         LinkConfig(n_prb=2, tb_bits=152, channel_model="awgn", csi="estimated"),
         [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
    ]


def selftest() -> bool:
    """
    input:
        none
    output:
        passed : bool
    """
    assert abs(qpsk_ber_theory(6.8) - 1e-3) / 1e-3 < 0.05
    assert abs(shannon_limit_biawgn_db(0.5) - 0.19) < 0.03
    assert abs(shannon_limit_complex_db(0.5) + 0.82) < 0.03

    lo, hi = wilson_interval(0, 100)
    assert np.isclose(lo, 0.0) and 0.0 < hi < 0.045
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0.0 and np.isclose(hi, 1.0 - 0.025 ** (1.0 / 100))
    assert abs(rule_of_three_upper(300) - 0.01) < 1e-12
    assert abs(goodput_mbps(76, 0.0, 0.5) - 0.152) < 1e-12

    cfg = LinkConfig(n_prb=2, tb_bits=76, channel_model="awgn", csi="estimated")
    low = simulate_point(cfg, -12.0, max_blocks=2, min_blocks=2, target_errors=4, seed=5)
    high = simulate_point(cfg, 12.0, max_blocks=2, min_blocks=2, target_errors=4, seed=6)
    assert low.bler >= high.bler and low.goodput_mbps <= high.goodput_mbps
    assert high.bler == 0.0 and np.isclose(high.goodput_mbps, 0.152)

    print("bler checks passed")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="PDSCH BLER and link performance")
    ap.add_argument("--full", action="store_true", help="simulate more blocks per point")
    ap.add_argument("--no-plot", action="store_true", help="skip figure generation")
    ap.add_argument("--selftest", action="store_true", help="run self-checks only")
    ap.add_argument("--replot", action="store_true", help="load saved CSV curves and regenerate plots")
    ap.add_argument("--phy", action="store_true", help="validate uncoded QPSK against theory")
    args = ap.parse_args()

    print("pdsch bler evaluation")

    if args.selftest:
        selftest()
        return

    if args.phy:
        rows = phy_validation(n_slots=40)
        if not args.no_plot:
            print(f"saved: {plot_phy_validation(rows)}")
        return
    if args.replot:
        csvs = sorted(DATA_DIR.glob("bler_curve_*.csv"))
        if not csvs:
            print(f"no CSV files found in {DATA_DIR}")
            return
        curves = [load_curve_csv(p) for p in csvs]
        print(f"loaded {len(curves)} curves from CSV")
        print_summary(curves)
        if not args.no_plot:
            for p in plot_curves(curves):
                print(f"saved: {p}")
            print(f"saved: {plot_ebn0_shannon(curves)}")
        return
    
    selftest()    



    if args.full:
        max_blocks, min_blocks, target_errors = 300, 40, 40
    else:
        max_blocks, min_blocks, target_errors = 60, 15, 20
    print(f"max_blocks={max_blocks}, min_blocks={min_blocks}, "
          f"target_errors={target_errors}")

    curves: List[CurveResult] = []
    t0 = time.perf_counter()
    for label, cfg, snr_list in preset_curves():
        curves.append(sweep(cfg, snr_list, label=label, max_blocks=max_blocks,
                            min_blocks=min_blocks, target_errors=target_errors,
                            seed=1000, verbose=True))
    print(f"elapsed: {time.perf_counter() - t0:.1f}s")

    print_summary(curves)

    for c in curves:
        print(f"saved: {save_curve_csv(c)}")

    same_bw = [c for c in curves if c.cfg.n_prb == 2 and c.cfg.channel_model == "awgn"]
    if len(same_bw) >= 2:
        for name, tgt in (("unconstrained", None),
                          ("BLER upper bound <= 0.2", 0.2)):

            print(f"link adaptation: {name}")

            amc = link_adaptation(same_bw, target_bler=tgt)
            print(f"  {'SNR(dB)':>8s} {'curve':>18s} {'rate':>6s} {'BLER':>6s} "
                  f"{'gp(Mbps)':>11s}")
            for row in amc:
                print(f"  {row['snr_db']:8.1f} {row['label']:>18s} {row['rate']:6.3f} "
                      f"{row['bler']:6.2f} {row['goodput_mbps']:11.3f}")

    if not args.no_plot:
        files = plot_curves(curves)
        files.append(plot_ebn0_shannon(curves))
        print("generated figures:")
        for p in files:
            print(f"  {p}")

    phy_rows = phy_validation(n_slots=20 if not args.full else 60)
    if not args.no_plot:
        print(f"saved: {plot_phy_validation(phy_rows)}")

    print("bler evaluation finished")


if __name__ == "__main__":
    main()
