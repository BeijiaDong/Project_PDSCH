import numpy as np

def ofdm_modulate(resource_grid, n_fft, cp_lengths, subcarrier_start=0):
    """
    input:
        resource_grid : ndarray, shape (n_sc, n_symbols)
        n_fft : int
        cp_lengths : list[int] | ndarray
        subcarrier_start : int
    output:
        time_signal : ndarray, shape (n_samples,)
    """
    N_sc, N_symbs = resource_grid.shape

    time_chunks = []

    for l in range(N_symbs):
        freq_domain = np.zeros(n_fft, dtype=complex)
        start = subcarrier_start
        end = start + N_sc
        freq_domain[start:end] = resource_grid[:, l]

        time_symbol = np.fft.ifft(freq_domain) * n_fft  # keep the transmit and receive scaling paired

        cp_len = cp_lengths[l]
        if cp_len > 0:
            cp_part = time_symbol[-cp_len:]  # copy the symbol tail
            symbol_with_cp = np.concatenate((cp_part, time_symbol))
        else:
            symbol_with_cp = time_symbol

        time_chunks.append(symbol_with_cp)

    time_signal = np.concatenate(time_chunks)
    return time_signal


def ofdm_demodulate(rx_signal, n_fft, cp_lengths, N_sc, subcarrier_start=0, symbol_offset=0):
    """
    input:
        rx_signal : ndarray, shape (n_samples,)
        n_fft : int
        cp_lengths : list[int] | ndarray
        N_sc : int
        subcarrier_start : int
        symbol_offset : int, sample index of the first symbol including its cp
    output:
        rx_grid : ndarray, shape (N_sc, n_symbols)
    """
    N_symbs = len(cp_lengths)
    rx_grid = np.zeros((N_sc, N_symbs), dtype=complex)

    idx = symbol_offset

    for l in range(N_symbs):
        cp_len = cp_lengths[l]

        idx_uf = idx + cp_len
        time_symbol = rx_signal[idx_uf : idx_uf + n_fft]

        freq_symbol = np.fft.fft(time_symbol) / n_fft

        start = subcarrier_start
        end = start + N_sc
        rx_grid[:, l] = freq_symbol[start:end]

        idx += cp_len + n_fft

    return rx_grid
