"""
(C) Copyright Daan Camps, Mercy Amankwah, E. Wes Bethel, Talita Perciano and
              Roel Van Beeumen

QPIXL - Quantum Pixel Library (Qiskit Python Implementation)
Utility functions for bit manipulation and power-of-2 operations
"""

import math


def ispow2(x: int) -> bool:
    """
    Return whether x is zero or a power of 2.
    
    Args:
        x: Integer to check
        
    Returns:
        True if x is 0 or a power of 2, False otherwise
    """
    return not (x & (x - 1))


def ilog2(x: int) -> int:
    """
    Returns the exponent of a power of 2.
    
    Args:
        x: Must be a power of 2
        
    Returns:
        The exponent n such that x = 2^n
        
    Raises:
        AssertionError if x is not a power of 2
    """
    assert ispow2(x) and x > 0, f"x={x} must be a power of 2"
    
    # Count trailing zeros (equivalent to std::countr_zero)
    if x == 0:
        return 0
    
    count = 0
    while (x & 1) == 0:
        x >>= 1
        count += 1
    
    return count


def nextpow2(x: int) -> int:
    """
    Return the next power of two greater than or equal to x.
    
    Args:
        x: Input integer
        
    Returns:
        The smallest power of 2 that is >= x
    """
    if x <= 0:
        return 1
    
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    x += 1
    
    return x


# Alternative implementations using Python's math library
def ilog2_alt(x: int) -> int:
    """
    Alternative implementation of ilog2 using bit_length.
    
    Args:
        x: Must be a power of 2
        
    Returns:
        The exponent n such that x = 2^n
    """
    assert ispow2(x) and x > 0, f"x={x} must be a power of 2"
    return x.bit_length() - 1


def nextpow2_alt(x: int) -> int:
    """
    Alternative implementation of nextpow2 using math.log2.
    
    Args:
        x: Input integer
        
    Returns:
        The smallest power of 2 that is >= x
    """
    if x <= 0:
        return 1
    return 2 ** math.ceil(math.log2(x))


# Utility function for Qiskit quantum computing
def num_qubits_for_size(size: int) -> int:
    """
    Calculate the number of qubits needed to represent a given size.
    
    Args:
        size: The size of the data
        
    Returns:
        Number of qubits needed (log2 of next power of 2)
    """
    return ilog2(nextpow2(size))


def num_qubits_for_image(nrows: int, ncols: int) -> int:
    """
    Calculate the number of qubits needed for a 2D image.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        
    Returns:
        Number of qubits needed for the image
    """
    return num_qubits_for_size(nrows * ncols)