import unittest
from gold_sequence import gold_sequence

class TestGold(unittest.TestCase):
    def test_length(self):
        for length in [1, 10, 100, 1000]:
            with self.subTest(length=length):
                self.assertEqual(len(gold_sequence(0, length)), length)

    def test_bits(self):
        seq = gold_sequence(0, 10000)
        self.assertTrue(all(b in (0,1) for b in seq))

    def test_different_cinit(self):
        seq0 = gold_sequence(0, 50)
        seq1 = gold_sequence(1, 50)
        self.assertNotEqual(seq0, seq1)

    def test_consistency(self):
        seq1 = gold_sequence(5, 100)
        seq2 = gold_sequence(5, 200)
        self.assertEqual(seq1, seq2[:100])

    def test_not_all_zero_not_all_one(self):
        seq = gold_sequence(0, 100)
        self.assertFalse(all(b == 0 for b in seq))
        self.assertFalse(all(b == 1 for b in seq))

if __name__ == '__main__':
    unittest.main()
