#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


LEVEL_TABLE: Dict[int, Tuple[int, ...]] = {  # bit patterns map to unnormalized axis levels
    2: (1, -1),
    4: (1, 3, -1, -3),
    6: (3, 1, 5, 7, -3, -1, -5, -7),
    8: (5, 7, 3, 1, 11, 9, 13, 15,
        -5, -7, -3, -1, -11, -9, -13, -15),
}
# scale the average symbol power to one
NORM_FACTOR: Dict[int, float] = {
    2: 1.0 / np.sqrt(2.0),
    4: 1.0 / np.sqrt(10.0),
    6: 1.0 / np.sqrt(42.0),
    8: 1.0 / np.sqrt(170.0),
}

QAM_NAME: Dict[int, str] = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}



def generate_constellation(qm: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    input:
        qm : int
    output:
        constellation : ndarray, unnormalized, shape (2**qm,)
        labels : ndarray, shape (2**qm,)
    """

    m = qm // 2
    levels = np.asarray(LEVEL_TABLE[qm], dtype=float)
    idx = np.arange(1 << qm) # the first bit is the most significant bit

    i_idx = np.zeros_like(idx)
    q_idx = np.zeros_like(idx)
    for k in range(m):
        i_idx |= ((idx >> (qm - 1 - 2 * k)) & 1) << (m - 1 - k)  # collect the even bit positions
        q_idx |= ((idx >> (qm - 1 - (2 * k + 1))) & 1) << (m - 1 - k)  # collect the odd bit positions

    constellation = levels[i_idx] + 1j * levels[q_idx]
    return constellation, idx


def generate_constellation_normalized(qm: int) -> np.ndarray:
    """
    input:
        qm : int
    output:
        constellation : ndarray, unit average power
    """
    constellation, _ = generate_constellation(qm)
    return constellation * NORM_FACTOR[qm]


def axis_levels(qm: int) -> np.ndarray:
    """
    input:
        qm : int
    output:
        levels : ndarray, shape (2**(qm // 2),)
    """

    return np.asarray(LEVEL_TABLE[qm], dtype=float)


def _gray_violations(qm: int) -> int:
    """
    input:
        qm : int
    output:
        count : int, directed neighbor pairs with invalid labels
    """
    levels = axis_levels(qm).astype(int)
    pos: Dict[Tuple[int, int], int] = {}
    for k in range(1 << qm):
        m = qm // 2
        ii = sum(((k >> (qm - 1 - 2 * t)) & 1) << (m - 1 - t) for t in range(m))
        qi = sum(((k >> (qm - 1 - (2 * t + 1))) & 1) << (m - 1 - t) for t in range(m))
        pos[(int(levels[ii]), int(levels[qi]))] = k

    bad = 0
    for (x, y), k in pos.items():
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nb = pos.get((x + dx, y + dy))
            if nb is not None and bin(k ^ nb).count("1") != 1:
                bad += 1
    return bad


def self_check(verbose: bool = True) -> bool:
    """
    input:
        verbose : bool
    output:
        passed : bool
    """
    ok = True
    for qm in sorted(LEVEL_TABLE):
        symbols = generate_constellation_normalized(qm)
        power = float(np.mean(np.abs(symbols) ** 2))
        gray_bad = _gray_violations(qm)

        passed = abs(power - 1.0) < 1e-12 and gray_bad == 0
        ok = ok and passed

        if verbose:
            status = "passed" if passed else "failed"
            print(
                f"{QAM_NAME[qm]}: power={power:.6f}, "
                f"gray errors={gray_bad}, {status}"
            )

    return ok


def verify_against_reference(csv_path: Optional[str] = None,
                             verbose: bool = True) -> bool:
    """
    input:
        csv_path : str | none
        verbose : bool
    output:
        passed : bool
    """
    if csv_path is None:
        csv_path = str(Path(__file__).parent / "qam_ref_38_211.csv")
    path = Path(csv_path)

    ref: Dict[int, Dict[int, Tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ref.setdefault(int(row["qm"]), {})[int(row["index"])] = (
                float(row["I"]), float(row["Q"]))

    ok_all = True
    if verbose:
        print(f"reference: {path.name}")
    for qm in sorted(ref):
        constellation, _ = generate_constellation(qm)
        bad: List[Tuple[int, complex, float, float]] = []
        for k, (ri, rq) in sorted(ref[qm].items()):
            if (abs(constellation[k].real - ri) > 1e-9
                    or abs(constellation[k].imag - rq) > 1e-9):
                bad.append((k, constellation[k], ri, rq))
        ok_all = ok_all and not bad
        if verbose:
            matched = len(ref[qm]) - len(bad)
            status = "passed" if not bad else "failed"
            print(
                f"{QAM_NAME[qm]}: "
                f"{matched}/{len(ref[qm])} matched, {status}"
            )
    return ok_all


def print_mapping(qm: int, limit: Optional[int] = None) -> None:
    """
    input:
        qm : int
        limit : int | none
    output:
        none
    """
    constellation, labels = generate_constellation(qm)
    step = 1 if limit is None else max(1, len(labels) // limit)

    print(f"{QAM_NAME[qm]} mapping")
    for label, point in zip(labels[::step], constellation[::step]):
        print(
            f"{label:0{qm}b} -> "
            f"({point.real:+.0f}, {point.imag:+.0f})"
        )


if __name__ == "__main__":
    assert self_check(verbose=False)
    assert verify_against_reference(verbose=False)
    print("all checks passed")
