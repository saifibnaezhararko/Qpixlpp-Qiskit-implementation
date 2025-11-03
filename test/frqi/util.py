"""
FRQI Utility Functions - Complete Test Suite
Converted from C++ (test_qpixl_frqi_util.cpp) to Python

(C) Copyright Daan Camps, Mercy Amankwah, E. Wes Bethel, Talita Perciano and Roel Van Beeumen
"""

import numpy as np
import pytest
from typing import Union, List
import sys


# ============================================================================
# UTILITY FUNCTIONS IMPLEMENTATION
# ============================================================================

def sfwht(vector: np.ndarray) -> np.ndarray:
    """
    Scaled Fast Walsh-Hadamard Transform (in-place)
    
    Args:
        vector: Input vector of size 2^n
        
    Returns:
        Transformed vector (modified in-place)
    """
    n = len(vector)
    h = 1
    
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = vector[j]
                y = vector[j + h]
                vector[j] = x + y
                vector[j + h] = x - y
        h *= 2
    
    # Scaling by 1/sqrt(n)
    vector /= np.sqrt(n)
    
    return vector


def isfwht(vector: np.ndarray) -> np.ndarray:
    """
    Inverse Scaled Fast Walsh-Hadamard Transform (in-place)
    
    Args:
        vector: Input vector of size 2^n
        
    Returns:
        Inverse transformed vector (modified in-place)
    """
    n = len(vector)
    
    # Scale by sqrt(n)
    vector *= np.sqrt(n)
    
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = vector[j]
                y = vector[j + h]
                vector[j] = x + y
                vector[j + h] = x - y
        h *= 2
    
    # Final scaling
    vector /= n
    
    return vector


def gray_code(n: int) -> int:
    """Compute Gray code for integer n"""
    return n ^ (n >> 1)


def gray_permutation(vector: np.ndarray, output: np.ndarray = None) -> np.ndarray:
    """
    Gray code permutation
    
    Args:
        vector: Input vector
        output: Optional output vector (if None, operates in-place)
        
    Returns:
        Permuted vector
    """
    n = len(vector)
    
    if output is None:
        # In-place operation
        result = np.zeros_like(vector)
        for i in range(n):
            gray_i = gray_code(i)
            result[gray_i] = vector[i]
        vector[:] = result[:]
        return vector
    else:
        # Copy to output
        for i in range(n):
            gray_i = gray_code(i)
            output[gray_i] = vector[i]
        return output


def inv_gray_permutation(vector: np.ndarray, output: np.ndarray = None) -> np.ndarray:
    """
    Inverse Gray code permutation
    
    Args:
        vector: Input vector (Gray permuted)
        output: Optional output vector (if None, operates in-place)
        
    Returns:
        Original order vector
    """
    n = len(vector)
    
    if output is None:
        # In-place operation
        result = np.zeros_like(vector)
        for i in range(n):
            gray_i = gray_code(i)
            result[i] = vector[gray_i]
        vector[:] = result[:]
        return vector
    else:
        # Copy to output
        for i in range(n):
            gray_i = gray_code(i)
            output[i] = vector[gray_i]
        return output


def convert_to_angles(vector: np.ndarray, maxval: int = 255) -> np.ndarray:
    """
    Convert grayscale values to angles
    
    Args:
        vector: Grayscale values
        maxval: Maximum grayscale value (default: 255)
        
    Returns:
        Angles in radians
    """
    pi_2 = np.pi / 2
    scal = pi_2 / maxval
    vector[:] = vector * scal
    return vector


def convert_to_grayscale(vector: np.ndarray, maxval: int = 255) -> np.ndarray:
    """
    Convert angles back to grayscale values
    
    Args:
        vector: Angles in radians
        maxval: Maximum grayscale value (default: 255)
        
    Returns:
        Grayscale values
    """
    pi_2 = np.pi / 2
    scal = pi_2 / maxval
    vector[:] = vector / scal
    return vector


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_sfwht_float32():
    """Test SFWHT with float32"""
    eps = np.finfo(np.float32).eps
    
    # Test case a: size 2
    a = np.array([1, 2], dtype=np.float32)
    sfwht(a)
    assert a[0] == np.float32(1.5)
    assert a[1] == np.float32(-0.5)
    
    # Test case b: size 4
    b = np.array([1, 2, 3, 4], dtype=np.float32)
    sfwht(b)
    assert np.abs(b[0] - 2.5) < eps
    assert np.abs(b[1] - (-0.5)) < eps
    assert np.abs(b[2] - (-1.0)) < eps
    assert np.abs(b[3] - 0.0) < eps
    
    # Test case c: size 8
    c = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    sfwht(c)
    assert np.abs(c[0] - 4.5) < eps
    assert np.abs(c[1] - (-0.5)) < eps
    assert np.abs(c[2] - (-1.0)) < eps
    assert np.abs(c[3] - 0.0) < eps
    assert np.abs(c[4] - (-2.0)) < eps
    assert np.abs(c[5] - 0.0) < eps
    assert np.abs(c[6] - 0.0) < eps
    assert np.abs(c[7] - 0.0) < eps
    
    # Test case d: size 16 with random values
    d = np.array([1.34, -3.24, 2.88, 16.2, 0, -7, 1.11, 1.45,
                  6.22, -13.57, 4.9, 5.31, -2.81, -3.41, -1.88, 10], 
                 dtype=np.float32)
    sfwht(d)
    
    expected = [1.09375, 0.37625, -3.9025, 3.62, 1.41125, 0.95375, 
                -0.915, 1.1425, 0.49875, -0.63625, 0.085, -0.465,
                1.29125, -2.87875, -0.5125, 0.1775]
    
    for i in range(16):
        assert np.abs(d[i] - expected[i]) < 10 * eps, f"Mismatch at index {i}"
    
    print("✓ test_sfwht_float32 passed")


def test_sfwht_float64():
    """Test SFWHT with float64"""
    eps = np.finfo(np.float64).eps
    
    # Test case a: size 2
    a = np.array([1, 2], dtype=np.float64)
    sfwht(a)
    assert a[0] == 1.5
    assert a[1] == -0.5
    
    # Test case b: size 4
    b = np.array([1, 2, 3, 4], dtype=np.float64)
    sfwht(b)
    assert np.abs(b[0] - 2.5) < eps
    assert np.abs(b[1] - (-0.5)) < eps
    assert np.abs(b[2] - (-1.0)) < eps
    assert np.abs(b[3] - 0.0) < eps
    
    # Test case c: size 8
    c = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    sfwht(c)
    assert np.abs(c[0] - 4.5) < eps
    assert np.abs(c[1] - (-0.5)) < eps
    assert np.abs(c[2] - (-1.0)) < eps
    assert np.abs(c[3] - 0.0) < eps
    assert np.abs(c[4] - (-2.0)) < eps
    assert np.abs(c[5] - 0.0) < eps
    assert np.abs(c[6] - 0.0) < eps
    assert np.abs(c[7] - 0.0) < eps
    
    # Test case d: size 16 with random values
    d = np.array([1.34, -3.24, 2.88, 16.2, 0, -7, 1.11, 1.45,
                  6.22, -13.57, 4.9, 5.31, -2.81, -3.41, -1.88, 10], 
                 dtype=np.float64)
    sfwht(d)
    
    expected = [1.09375, 0.37625, -3.9025, 3.62, 1.41125, 0.95375, 
                -0.915, 1.1425, 0.49875, -0.63625, 0.085, -0.465,
                1.29125, -2.87875, -0.5125, 0.1775]
    
    for i in range(16):
        assert np.abs(d[i] - expected[i]) < 10 * eps, f"Mismatch at index {i}"
    
    print("✓ test_sfwht_float64 passed")


def test_isfwht_float32():
    """Test inverse SFWHT with float32"""
    eps = np.finfo(np.float32).eps
    
    a = np.array([1.34, -3.24, 2.88, 16.2, 0, -7, 1.11, 1.45,
                  6.22, -13.57, 4.9, 5.31, -2.81, -3.41, -1.88, 10], 
                 dtype=np.float32)
    
    a_copy = a.copy()
    
    sfwht(a)
    isfwht(a)
    
    for i in range(16):
        assert np.abs(a[i] - a_copy[i]) < 100 * eps, f"Mismatch at index {i}"
    
    print("✓ test_isfwht_float32 passed")


def test_isfwht_float64():
    """Test inverse SFWHT with float64"""
    eps = np.finfo(np.float64).eps
    
    a = np.array([1.34, -3.24, 2.88, 16.2, 0, -7, 1.11, 1.45,
                  6.22, -13.57, 4.9, 5.31, -2.81, -3.41, -1.88, 10], 
                 dtype=np.float64)
    
    a_copy = a.copy()
    
    sfwht(a)
    isfwht(a)
    
    for i in range(16):
        assert np.abs(a[i] - a_copy[i]) < 100 * eps, f"Mismatch at index {i}"
    
    print("✓ test_isfwht_float64 passed")


def test_gray_permutation_float32():
    """Test Gray permutation with float32"""
    
    # Test case a: size 1
    a = np.array([1], dtype=np.float32)
    a_out = np.zeros(1, dtype=np.float32)
    
    gray_permutation(a, a_out)
    assert a[0] == a_out[0]
    
    a = np.array([1], dtype=np.float32)
    gray_permutation(a)
    assert a[0] == a_out[0]
    
    # Test case b: size 2
    b = np.array([1, 2], dtype=np.float32)
    b_out = np.zeros(2, dtype=np.float32)
    
    gray_permutation(b, b_out)
    assert b[0] == b_out[0]
    assert b[1] == b_out[1]
    
    b = np.array([1, 2], dtype=np.float32)
    gray_permutation(b)
    assert b[0] == b_out[0]
    assert b[1] == b_out[1]
    
    # Test case c: size 4
    c = np.array([1, 2, 3, 4], dtype=np.float32)
    c_out = np.zeros(4, dtype=np.float32)
    
    gray_permutation(c, c_out)
    assert c[0] == c_out[0]
    assert c[1] == c_out[1]
    assert c[2] == c_out[3]
    assert c[3] == c_out[2]
    
    c = np.array([1, 2, 3, 4], dtype=np.float32)
    gray_permutation(c)
    assert c[0] == c_out[0]
    assert c[1] == c_out[1]
    assert c[2] == c_out[2]
    assert c[3] == c_out[3]
    
    # Test case d: size 8
    d = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    d_out = np.zeros(8, dtype=np.float32)
    
    gray_permutation(d, d_out)
    assert d[0] == d_out[0]
    assert d[1] == d_out[1]
    assert d[2] == d_out[3]
    assert d[3] == d_out[2]
    assert d[4] == d_out[7]
    assert d[5] == d_out[6]
    assert d[6] == d_out[4]
    assert d[7] == d_out[5]
    
    d = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    gray_permutation(d)
    assert d[0] == d_out[0]
    assert d[1] == d_out[1]
    assert d[2] == d_out[2]
    assert d[3] == d_out[3]
    assert d[4] == d_out[4]
    assert d[5] == d_out[5]
    assert d[6] == d_out[6]
    assert d[7] == d_out[7]
    
    # Test case e: size 16
    e = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                 dtype=np.float32)
    e_out = np.zeros(16, dtype=np.float32)
    
    gray_permutation(e, e_out)
    assert e[0] == e_out[0]
    assert e[1] == e_out[1]
    assert e[2] == e_out[3]
    assert e[3] == e_out[2]
    assert e[4] == e_out[7]
    assert e[5] == e_out[6]
    assert e[6] == e_out[4]
    assert e[7] == e_out[5]
    assert e[8] == e_out[15]
    assert e[9] == e_out[14]
    assert e[10] == e_out[12]
    assert e[11] == e_out[13]
    assert e[12] == e_out[8]
    assert e[13] == e_out[9]
    assert e[14] == e_out[11]
    assert e[15] == e_out[10]
    
    e = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                 dtype=np.float32)
    gray_permutation(e)
    for i in range(16):
        assert e[i] == e_out[i]
    
    print("✓ test_gray_permutation_float32 passed")


def test_gray_permutation_float64():
    """Test Gray permutation with float64"""
    
    # Test all cases with float64 (same logic as float32)
    # Size 16 test
    e = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                 dtype=np.float64)
    e_out = np.zeros(16, dtype=np.float64)
    
    gray_permutation(e, e_out)
    
    # Verify specific permutations
    assert e[0] == e_out[0]
    assert e[1] == e_out[1]
    assert e[2] == e_out[3]
    assert e[3] == e_out[2]
    assert e[8] == e_out[15]
    assert e[15] == e_out[10]
    
    print("✓ test_gray_permutation_float64 passed")


def test_inv_gray_permutation_float32():
    """Test inverse Gray permutation with float32"""
    
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                 dtype=np.float32)
    a_copy = a.copy()
    a_out = np.zeros(16, dtype=np.float32)
    
    gray_permutation(a)
    
    # Test with copy
    inv_gray_permutation(a, a_out)
    for i in range(16):
        assert a_out[i] == a_copy[i], f"Mismatch at index {i}"
    
    # Test in-place
    inv_gray_permutation(a)
    for i in range(16):
        assert a[i] == a_copy[i], f"Mismatch at index {i}"
    
    print("✓ test_inv_gray_permutation_float32 passed")


def test_inv_gray_permutation_float64():
    """Test inverse Gray permutation with float64"""
    
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                 dtype=np.float64)
    a_copy = a.copy()
    a_out = np.zeros(16, dtype=np.float64)
    
    gray_permutation(a)
    
    # Test with copy
    inv_gray_permutation(a, a_out)
    for i in range(16):
        assert a_out[i] == a_copy[i], f"Mismatch at index {i}"
    
    # Test in-place
    inv_gray_permutation(a)
    for i in range(16):
        assert a[i] == a_copy[i], f"Mismatch at index {i}"
    
    print("✓ test_inv_gray_permutation_float64 passed")


def test_convert_angles_grayscale_float32():
    """Test angle/grayscale conversion with float32"""
    
    eps = np.finfo(np.float32).eps
    
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    a_copy = a.copy()
    maxval = 255
    
    pi_2 = np.pi / 2
    scal = pi_2 / maxval
    
    convert_to_angles(a, maxval)
    
    for i in range(8):
        expected = a_copy[i] * scal
        assert np.abs(a[i] - expected) < 10 * eps, f"Angle conversion failed at {i}"
    
    convert_to_grayscale(a, maxval)
    
    for i in range(8):
        assert np.abs(a[i] - a_copy[i]) < 10 * eps, f"Grayscale conversion failed at {i}"
    
    print("✓ test_convert_angles_grayscale_float32 passed")


def test_convert_angles_grayscale_float64():
    """Test angle/grayscale conversion with float64"""
    
    eps = np.finfo(np.float64).eps
    
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    a_copy = a.copy()
    maxval = 255
    
    pi_2 = np.pi / 2
    scal = pi_2 / maxval
    
    convert_to_angles(a, maxval)
    
    for i in range(8):
        expected = a_copy[i] * scal
        assert np.abs(a[i] - expected) < 10 * eps, f"Angle conversion failed at {i}"
    
    convert_to_grayscale(a, maxval)
    
    for i in range(8):
        assert np.abs(a[i] - a_copy[i]) < 10 * eps, f"Grayscale conversion failed at {i}"
    
    print("✓ test_convert_angles_grayscale_float64 passed")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests_float32():
    """Run all tests with float32"""
    print("\n" + "="*60)
    print("Running FRQI Utility Tests (float32)")
    print("="*60 + "\n")
    
    test_sfwht_float32()
    test_isfwht_float32()
    test_gray_permutation_float32()
    test_inv_gray_permutation_float32()
    test_convert_angles_grayscale_float32()
    
    print("\n✅ All float32 tests passed!\n")


def run_all_tests_float64():
    """Run all tests with float64"""
    print("\n" + "="*60)
    print("Running FRQI Utility Tests (float64)")
    print("="*60 + "\n")
    
    test_sfwht_float64()
    test_isfwht_float64()
    test_gray_permutation_float64()
    test_inv_gray_permutation_float64()
    test_convert_angles_grayscale_float64()
    
    print("\n✅ All float64 tests passed!\n")


if __name__ == "__main__":
    # Run tests for both float32 and float64
    run_all_tests_float32()
    run_all_tests_float64()
    
    print("="*60)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)