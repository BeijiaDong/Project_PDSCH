import numpy as np
from modulatordemodulator import ModulatorDemodulator
from qam_constellation import verify_against_reference


def test_qpsk_modulation():
    mod = ModulatorDemodulator(2)
    bits = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)

    # normalize the four qpsk points
    expected = np.array([
        1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j
    ]) / np.sqrt(2)

    symbols = mod.modulate(bits)
    np.testing.assert_allclose(symbols, expected, rtol=0, atol=1e-12)


def test_zero_noise_demodulation():
    rng = np.random.default_rng(42)

    for qm in [2, 4, 6, 8]:
        mod = ModulatorDemodulator(qm)
        bits = rng.integers(0, 2, size=100 * qm, dtype=np.uint8)
        symbols = mod.modulate(bits)

        # avoid division by zero in the llr calculation
        llrs = mod.demodulate(symbols, noise_var=1e-10)
        decisions = (llrs < 0).astype(np.uint8)

        np.testing.assert_array_equal(decisions, bits)


def test_constellation_matches_reference():
    assert verify_against_reference(verbose=False), "reference mismatch"
    print("[OK] constellation table matches MATLAB reference")


if __name__ == "__main__":
    test_qpsk_modulation()
    test_zero_noise_demodulation()
    test_constellation_matches_reference()
    print("All tests passed!")
