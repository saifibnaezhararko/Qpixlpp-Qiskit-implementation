"""

eta te ektu issue ase



FRQI (Flexible Representation of Quantum Images) Utility Functions

Converted from C++ implementation by Daan Camps, Mercy Amankwah, 
E. Wes Bethel, Talita Perciano and Roel Van Beeumen
"""

import numpy as np
from typing import Optional


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def ilog2(n: int) -> int:
    """Compute the integer log base 2 of n."""
    return int(np.log2(n))


def sfwht(a: np.ndarray) -> None:
    """
    Computes a scaled fast Walsh-Hadamard transform in binary ordering.
    
    The size of the input vector should be a power of 2.
    This function modifies the input array in-place.
    
    Args:
        a: Input array (modified in-place)
    """
    n = len(a)
    assert is_power_of_2(n), "Array size must be a power of 2"
    
    j = 1
    while j < n:
        for i in range(n):
            if (i & j) == 0:
                j1 = i + j
                x = a[i]
                y = a[j1]
                
                a[i] = (x + y) / 2.0
                a[j1] = (x - y) / 2.0
        j *= 2


def isfwht(a: np.ndarray) -> None:
    """
    In-place computation of an inverse scaled fast Walsh-Hadamard 
    transform in binary order of an input vector.
    
    The size of the input vector should be a power of 2.
    
    Args:
        a: Input array (modified in-place)
    """
    n = len(a)
    assert is_power_of_2(n), "Array size must be a power of 2"
    
    j = 1
    while j < n:
        for i in range(n):
            if (i & j) == 0:
                j1 = i + j
                x = a[i]
                
                a[i] = a[i] + a[j1]
                a[j1] = x - a[j1]
        j *= 2


def gray_code(x: int) -> int:
    """Compute the Gray code of x."""
    return x ^ (x >> 1)


def gray_permutation(a: np.ndarray, b: Optional[np.ndarray] = None) -> None:
    """
    Permutes array by Gray code permutation.
    
    If b is provided: Permutes a into b (a unchanged, result in b)
    If b is None: Permutes a in-place using optimized algorithm
    
    Args:
        a: Input array
        b: Output array (optional)
    """
    n = len(a)
    
    if b is not None:
        # Two-argument version: permute a into b
        assert len(b) == n, "Arrays must have the same size"
        for k in range(n):
            b[k] = a[gray_code(k)]
    else:
        # In-place version (more complex algorithm)
        z = 1
        u = 0
        v = 0
        cl = 1
        
        ldm = 1
        m = 2
        while m < n:
            z <<= 1
            v <<= 1
            if is_power_of_2(ldm):
                z += 1
                cl <<= 1
            else:
                v += 1
            
            u = 0
            while True:
                u = (u - v) & v
                i = z | u
                t = a[i]
                g = gray_code(i)
                k = cl - 1
                while k != 0:
                    a[i] = a[g]
                    i = g
                    g = gray_code(i)
                    k -= 1
                a[i] = t
                
                if u == 0:
                    break
            
            ldm += 1
            m <<= 1


def inv_gray_permutation(a: np.ndarray, b: Optional[np.ndarray] = None) -> None:
    """
    Permutes array by inverse Gray code permutation.
    
    If b is provided: Permutes a into b (a unchanged, result in b)
    If b is None: Permutes a in-place using optimized algorithm
    
    Args:
        a: Input array
        b: Output array (optional)
    """
    n = len(a)
    
    if b is not None:
        # Two-argument version: permute a into b
        assert len(b) == n, "Arrays must have the same size"
        for k in range(n):
            b[gray_code(k)] = a[k]
    else:
        # In-place version (more complex algorithm)
        z = 1
        u = 0
        v = 0
        cl = 1
        
        ldm = 1
        m = 2
        while m < n:
            z <<= 1
            v <<= 1
            if is_power_of_2(ldm):
                z += 1
                cl <<= 1
            else:
                v += 1
            
            u = 0
            while True:
                u = (u - v) & v
                i = z | u
                t = a[i]
                g = gray_code(i)
                k = cl - 1
                while k != 0:
                    tt = a[g]
                    a[g] = t
                    t = tt
                    g = gray_code(g)
                    k -= 1
                a[g] = t
                
                if u == 0:
                    break
            
            ldm += 1
            m <<= 1


def convert_to_angles(a: np.ndarray, maxval: int) -> None:
    """
    Convert a vector containing grayscale data to angles for FRQI.
    
    Args:
        a: Input array containing grayscale values (modified in-place)
        maxval: Maximum grayscale value
    """
    pi2 = np.pi / 2  # 2 * arctan(1) = pi/2
    scal = pi2 / maxval
    a *= scal


def convert_to_grayscale(a: np.ndarray, maxval: int) -> None:
    """
    Convert a vector containing angles for FRQI to grayscale data.
    
    Args:
        a: Input array containing angle values (modified in-place)
        maxval: Maximum grayscale value to scale to
    """
    pi2 = np.pi / 2  # 2 * arctan(1) = pi/2
    scal = maxval / pi2
    a *= scal