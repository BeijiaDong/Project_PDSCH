def gold_sequence(c_init: int, length: int) -> list[int]:
    """
    input:
        c_init : int, 31-bit unsigned seed 
        length : int, non-negative bit count
    output:
        gold : list[int], length bits
    """
    N_c = 1600  # skip the first 1600 register outputs
    total_len = N_c + length

    x1 = [0] * total_len
    x2 = [0] * total_len

    x1[0] = 1

    for i in range(31):  # seed the second register
        x2[i] = (c_init >> i) & 1

    for n in range(total_len - 31):
        x1[n + 31] = (x1[n + 3] + x1[n]) % 2
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) % 2

    gold = [0] * length
    for k in range(length):
        idx = N_c + k
        gold[k] = (x1[idx] + x2[idx]) % 2

    return gold
