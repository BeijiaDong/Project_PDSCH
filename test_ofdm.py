import numpy as np
from ofdm import ofdm_modulate, ofdm_demodulate

N_sc = 240
N_symbs = 14
n_fft = 1024
cp_lengths = [88] + [72]*13


np.random.seed(42)
tx_grid = (np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], (N_sc, N_symbs)) / np.sqrt(2))

tx_signal = ofdm_modulate(tx_grid, n_fft, cp_lengths)
expected_len = 14*1024 + sum(cp_lengths)
assert len(tx_signal) == expected_len, "unexpected signal length"

rx_grid = ofdm_demodulate(tx_signal, n_fft, cp_lengths, N_sc)

error = np.max(np.abs(tx_grid - rx_grid))
assert error < 1e-12, "ofdm roundtrip failed"
print(f"roundtrip passed: error={error:.2e}")

tx_grid2 = np.zeros((N_sc, N_symbs), dtype=complex)
tx_grid2[0, :] = 1.0 + 0.0j

tx_signal2 = ofdm_modulate(tx_grid2, n_fft, cp_lengths)

assert np.allclose(tx_signal2, 1.0 + 0.0j), "unexpected dc waveform"

rx_grid2 = ofdm_demodulate(tx_signal2, n_fft, cp_lengths, N_sc)
error2 = np.max(np.abs(tx_grid2 - rx_grid2))
assert error2 < 1e-12, "dc roundtrip failed"
print(f"dc test passed: error={error2:.2e}")

tx_grid3 = np.random.randn(N_sc, N_symbs) + 1j*np.random.randn(N_sc, N_symbs)
tx_signal3_clean = ofdm_modulate(tx_grid3, n_fft, cp_lengths)

tx_signal3_corrupted = tx_signal3_clean.copy()
tx_signal3_corrupted[:75] = 999.9 + 999.9j  # change samples inside the first cyclic prefix

rx_clean = ofdm_demodulate(tx_signal3_clean, n_fft, cp_lengths, N_sc)
rx_corrupted = ofdm_demodulate(tx_signal3_corrupted, n_fft, cp_lengths, N_sc)

diff = np.max(np.abs(rx_clean - rx_corrupted))
assert diff < 1e-12, "cp corruption changed the recovered grid"
print(f"cp test passed: diff={diff:.2e}")


