import numpy as np
from qiskit import QuantumCircuit


def gray_code(n):
    """Compute Gray code of n"""
    return n ^ (n >> 1)


def gray_permutation(a):
    """Apply Gray permutation to vector a (in-place)"""
    n = len(a)
    result = np.zeros_like(a)
    for i in range(n):
        result[i] = a[gray_code(i)]
    a[:] = result


def sfwht(a):
    """Scaled fast Walsh-Hadamard transform (in-place)"""
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    # Scale by 1/n
    a /= n


def is_power_of_2(n):
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0


def ilog2(n):
    """Compute integer log base 2"""
    return int(np.log2(n))


def count_trailing_zeros(n):
    """Count trailing zeros in binary representation"""
    if n == 0:
        return 0
    count = 0
    while (n & 1) == 0:
        n >>= 1
        count += 1
    return count


def compressed_frqi_circuit(a, compression=0.0):
    """
    Generates a compressed FRQI circuit for the input vector `a`
    
    Parameters:
    -----------
    a : array_like
        Angle representation of the image, its size should be a power of 2.
        Note: This function modifies the input array.
    compression : float
        Compression value between 0 and 100 where:
        - 0 means no compression (all image coefficients are encoded)
        - 100 means full compression (no image coefficients are encoded)
    
    Returns:
    --------
    QuantumCircuit
        The compressed FRQI quantum circuit
    """
    # Convert to numpy array if not already
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=float)
    
    n = len(a)
    assert is_power_of_2(n), "Size of input vector must be a power of 2"
    
    k = ilog2(n)
    
    # Multiply angles by two
    a *= 2
    
    # Convert angles through a scaled permuted fast Walsh-Hadamard transform
    sfwht(a)
    gray_permutation(a)
    
    # Create index array [0, 1, 2, ..., n-1]
    index = list(range(n))
    
    # Sort vector a by absolute values and get ordering in index array
    index.sort(key=lambda i: abs(a[i]))
    
    # Set smallest absolute values of a to zero according to compression param
    cutoff = int((compression / 100.0) * n)
    for idx in index[:cutoff]:
        a[idx] = 0.0
    
    # Construct FRQI circuit
    # k qubits for position encoding + 1 qubit for pixel value
    circuit = QuantumCircuit(k + 1)
    
    # Hadamard register - apply to first k qubits
    for i in range(k):
        circuit.h(i)
    
    # Compressed uniformly controlled rotation register
    i = 0
    while i < (1 << k):  # 2^k
        # Reset parity check
        pc = 0
        
        # Add RY gate
        if a[i] != 0:
            circuit.ry(a[i], k)
        
        # Loop over sequence of consecutive zero angles
        while True:
            # Compute control qubit
            if i == (1 << k) - 1:
                ctrl = 0
            else:
                gray_xor = gray_code(i) ^ gray_code(i + 1)
                # Verify exactly one bit is set
                assert bin(gray_xor).count('1') == 1
                # Count trailing zeros and compute control
                ctrl = k - count_trailing_zeros(gray_xor) - 1
            
            # Update parity check
            pc ^= 1 << ctrl
            
            i += 1
            
            if not (i < (1 << k) and a[i] == 0):
                break
        
        # Add CNOTs determined by parity check
        for j in range(k):
            if (pc >> j) & 1:
                circuit.cx(j, k)
    
    return circuit


# Example usage:
if __name__ == "__main__":
    # Create a simple 4-element angle vector
    angles = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Generate FRQI circuit with 50% compression
    qc = compressed_frqi_circuit(angles.copy(), compression=50.0)
    
    print(qc)
    print(f"Circuit depth: {qc.depth()}")
    print(f"Number of gates: {len(qc)}")