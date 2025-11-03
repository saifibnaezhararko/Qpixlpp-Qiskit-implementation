import unittest


# Utility functions
def ilog2(n):
    """Integer log base 2"""
    if n <= 0:
        raise ValueError("ilog2 only defined for positive integers")
    return n.bit_length() - 1


def ispow2(n):
    """Check if n is a power of 2"""
    if n == 0:
        return True
    return n > 0 and (n & (n - 1)) == 0


def nextpow2(n):
    """Return the next power of 2 >= n"""
    if n <= 0:
        return 1
    if ispow2(n):
        return n
    return 1 << n.bit_length()


class TestQpixlUtil(unittest.TestCase):
    
    def test_ilog2(self):
        self.assertEqual(ilog2(1), 0)
        self.assertEqual(ilog2(2), 1)
        self.assertEqual(ilog2(4), 2)
        self.assertEqual(ilog2(8), 3)
        self.assertEqual(ilog2(16), 4)
        self.assertEqual(ilog2(32), 5)
        self.assertEqual(ilog2(64), 6)
    
    def test_ispow2(self):
        self.assertEqual(ispow2(0), True)
        self.assertEqual(ispow2(1), True)
        self.assertEqual(ispow2(2), True)
        self.assertEqual(ispow2(3), False)
        self.assertEqual(ispow2(1024), True)
        self.assertEqual(ispow2(1022), False)
    
    def test_nextpow2(self):
        self.assertEqual(nextpow2(1), 1)
        self.assertEqual(nextpow2(2), 2)
        self.assertEqual(nextpow2(3), 4)
        self.assertEqual(nextpow2(4), 4)
        self.assertEqual(nextpow2(6), 8)
        self.assertEqual(nextpow2(8), 8)
        self.assertEqual(nextpow2(100), 128)
        self.assertEqual(nextpow2(2000), 2048)


if __name__ == '__main__':
    unittest.main()