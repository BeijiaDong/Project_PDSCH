def crc24a_encoder(data_bits: list[int]) -> list[int]:
    """
    input:
        data_bits : list[int], binary bits
    output:
        data_bits + crc_bits: list[int], data followed by 24 crc bits
    """
  
    poly = 0x1864CFB # crc24a polynomial
    crc = 0  # 24-bit register
    
    for bit in data_bits:
        msb = (crc >> 23) & 1                # check bit 24 before shifting
        crc = ((crc << 1) | bit) & 0xFFFFFF  # shift left and shift in new bits
        if msb:                              # if msb is 1, then XOR with the polynomial
            crc ^= (poly & 0xFFFFFF)      

    # append 24 zeros before reading the remainder
    for _ in range(24):
        msb = (crc >> 23) & 1
        crc = ((crc << 1) | 0) & 0xFFFFFF   # shift left and shift in 0
        if msb:
            crc ^= (poly & 0xFFFFFF)
    

    crc_inv = (~crc) & 0xFFFFFF # invert the bits to obtain the check digit
    crc_bits = [(crc_inv >> (23 - i)) & 1 for i in range(24)]   # write the crc bits most significant first
    
    return data_bits + crc_bits

def crc24a_check(received_bits: list[int]) -> tuple[list[int], bool]:
    """
    input:
        received_bits : list[int], data bits with crc
    output:
        (data_bits, pass) :
            data_bits : list[int]
            pass : bool      
    """
    
    if len(received_bits) < 24:
         return [], False
    
    # divide the received data and crc together

    poly = 0x1864CFB # generator polynomial
    crc = 0  # 24-bit register
    for bit in received_bits:
            msb = (crc >> 23) & 1                
            crc = ((crc << 1) | bit) & 0xFFFFFF  
            if msb:                 
                crc ^= (poly & 0xFFFFFF)        
    pass_flag = (crc == 0xFFFFFF) # if crc is full 1, crc is passed 
    data_bits = received_bits[:-24] # only data bits
    return data_bits, pass_flag
