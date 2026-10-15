import numpy as np

from qam_constellation import generate_constellation, NORM_FACTOR

class ModulatorDemodulator:
    def __init__(self, mod_order: int):
        if (
            not isinstance(mod_order, (int, np.integer))
            or mod_order not in NORM_FACTOR
        ):
            raise ValueError("mod_order must be 2, 4, 6 or 8")

        self.Qm = int(mod_order)
        self.constellation, self.bit_labels = generate_constellation(self.Qm)
        self.norm_factor = float(NORM_FACTOR[self.Qm])


    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        input:
            bits : ndarray
        output:
            symbols : ndarray, shape (len(bits) // self.Qm,)
        """
        ...
        if len(bits) % self.Qm != 0:
            raise ValueError(
                f"bit count {bits.size} must be divisible by Qm={self.Qm}"
            )

        bit_groups = bits.reshape(-1, self.Qm)

        weights = 2 ** np.arange(self.Qm - 1, -1, -1)
        bit_indices = bit_groups.dot(weights)  # read each bit group as a binary index

        symbols = self.constellation[bit_indices]
        # normalize the average constellation power
        symbols = symbols * self.norm_factor

        return symbols

    def demodulate(self, rx_syms: np.ndarray,
                   noise_var: float,
                   channel: np.ndarray | None = None) -> np.ndarray:
        """
        input:
            rx_syms : ndarray
            noise_var : float
            channel : ndarray
        output:
            llrs : ndarray, positive for bit 0
        """
        ...
        N = len(rx_syms)

        if channel is None:
            channel = np.ones(N, dtype=np.complex128)
        elif np.isscalar(channel):
            channel = np.full(N, channel, dtype=np.complex128)


        const_norm = self.constellation * self.norm_factor


        rx_syms_expanded = rx_syms.reshape(-1, 1)
        channel_expanded = channel.reshape(-1, 1)
        const_expanded = const_norm.reshape(1, -1)
        # compare against channel-scaled constellation points
        distances = np.abs(rx_syms_expanded - channel_expanded * const_expanded) ** 2  # include the channel gain in each candidate symbol

        bit_labels_ordered = np.asarray(self.bit_labels)

        llrs_list = []
        for bit_pos in range(self.Qm):
            shift = self.Qm - 1 - bit_pos
            mask = (bit_labels_ordered >> shift) & 1
            # separate the candidates for bit 0 and bit 1
            idx0 = np.where(mask == 0)[0]
            idx1 = np.where(mask == 1)[0]

            dist0_min = np.min(distances[:, idx0], axis=1)
            dist1_min = np.min(distances[:, idx1], axis=1)

            llr = (dist1_min - dist0_min) / noise_var  # positive values favor bit 0

            llrs_list.append(llr)
        # keep the bit order within each received symbol
        llrs = np.column_stack(llrs_list).flatten()


        return llrs

