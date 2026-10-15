from CRC24A import crc24a_encoder
from CRC24A import crc24a_check


input1 = [0]
output1 = crc24a_encoder(input1)
expected1 = [0] + [1] * 24
print("test 1 (input [0]):")
print(f"  output:   {output1}")
print(f"  expected: {expected1}")
print(f"  [OK]" if output1 == expected1 else f"  [FAIL]")

print()

input2 = [1]
output2 = crc24a_encoder(input2)
expected_crc2 = [0,1,1,1, 1,0,0,1, 1,0,1,1, 0,0,1,1, 0,0,0,0, 0,1,0,0]
expected2 = [1] + expected_crc2
print("test 2 (input [1]):")
print(f"  output:   {output2}")
print(f"  expected: {expected2}")
print(f"  [OK]" if output2 == expected2 else f"  [FAIL]")

data = [1, 0, 1, 1, 1, 0]
encoded = crc24a_encoder(data)
rx_data, crc_ok = crc24a_check(encoded)
print("test 2.1 (no errors):")
print(f"  original data: {data}")
print(f"  encoded bits:  {encoded}")
print(f"  received data: {rx_data}")
print(f"  crc passed:    {crc_ok}")
print(f"  [OK]" if (crc_ok and rx_data == data) else f"  [FAIL]")

print()

encoded_err = encoded.copy()
encoded_err[0] ^= 1  # flip one bit to check error detection
rx_data_err, crc_ok_err = crc24a_check(encoded_err)
print("test 2.2 (single-bit error):")
print(f"  corrupted bits: {encoded_err}")
print(f"  crc passed:     {crc_ok_err}")
print(f"  [OK]" if (not crc_ok_err) else f"  [FAIL]")
