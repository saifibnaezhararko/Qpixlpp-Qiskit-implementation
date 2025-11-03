#!/usr/bin/env python3
"""
QPIXL Qiskit: Quantum circuits for FRQI image representation
Converted from QPIXL++ (C++) to Qiskit (Python)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RYGate
import argparse
import sys

def read_pgm_ascii(filename):
    """Read ASCII PGM file"""
    with open(filename, 'r') as f:
        magic = f.readline().strip()
        if magic != 'P2':
            raise ValueError("Not an ASCII PGM file")
        
        # Skip comments
        line = f.readline().strip()
        while line.startswith('#'):
            line = f.readline().strip()
        
        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().strip())
        
        data = []
        for line in f:
            data.extend(map(float, line.split()))
    
    return np.array(data), nrows, ncols, maxval

def write_pgm_ascii(filename, data, nrows, ncols, maxval, zero_pad=0):
    """Write ASCII PGM file"""
    actual_size = nrows * ncols
    data = data[:actual_size]
    
    with open(filename, 'w') as f:
        f.write('P2\n')
        f.write(f'{ncols} {nrows}\n')
        f.write(f'{int(maxval)}\n')
        
        for i, val in enumerate(data):
            f.write(f'{int(np.clip(val, 0, maxval))} ')
            if (i + 1) % ncols == 0:
                f.write('\n')
    return 0

def zero_pad_to_power_of_2(data):
    """Pad data to next power of 2"""
    size = len(data)
    next_pow2 = int(2 ** np.ceil(np.log2(size)))
    
    if size < next_pow2:
        padded = np.zeros(next_pow2)
        padded[:size] = data
        return padded
    return data

def convert_to_angles(data, maxval):
    """Convert grayscale values to FRQI angles"""
    return (data / maxval) * (np.pi / 2)

def convert_to_grayscale(angles, maxval):
    """Convert angles back to grayscale values"""
    return (angles / (np.pi / 2)) * maxval

def fwht(data):
    """Fast Walsh-Hadamard Transform"""
    data = data.copy()
    n = len(data)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = data[j]
                y = data[j + h]
                data[j] = x + y
                data[j + h] = x - y
        h *= 2
    return data

def isfwht(data):
    """Inverse Fast Walsh-Hadamard Transform"""
    return fwht(data)  # FWHT is self-inverse

def inv_gray_permutation(data):
    """Inverse of Gray code permutation"""
    n = len(data)
    temp = data.copy()
    for i in range(n):
        # Convert from Gray code to binary
        gray = i
        binary = 0
        while gray:
            binary ^= gray
            gray >>= 1
        if binary < n:
            data[i] = temp[binary]
    return data

def create_compressed_frqi_circuit(angles, compression=0.0):
    """Create compressed FRQI quantum circuit"""
    n_pixels = len(angles)
    n_position_qubits = int(np.log2(n_pixels))
    n_qubits = n_position_qubits + 1  # +1 for intensity qubit
    
    # Apply FWHT
    angles_fwht = fwht(angles)
    
    # Apply compression threshold
    if compression > 0:
        threshold = np.percentile(np.abs(angles_fwht), compression)
        angles_fwht[np.abs(angles_fwht) < threshold] = 0
    
    qc = QuantumCircuit(n_qubits)
    
    # Hadamard gates on position qubits
    for i in range(1, n_qubits):
        qc.h(i)
    
    # Encode controlled rotations
    for pos in range(n_pixels):
        angle = angles_fwht[pos]
        
        if abs(angle) < 1e-10:  # Skip due to compression
            continue
        
        # Binary representation
        binary = format(pos, f'0{n_position_qubits}b')
        
        # Multi-controlled RY gate
        control_state = binary[::-1]  # Reverse for qubit ordering
        
        # Add X gates for 0 bits
        for bit_idx, bit in enumerate(control_state):
            if bit == '0':
                qc.x(bit_idx + 1)
        
        # Controlled-RY rotation
        control_qubits = list(range(1, n_qubits))
        qc.append(RYGate(2 * angle).control(len(control_qubits)), 
                  control_qubits + [0])
        
        # Remove X gates
        for bit_idx, bit in enumerate(control_state):
            if bit == '0':
                qc.x(bit_idx + 1)
    
    return qc

def count_circuit_gates(circuit):
    """Count gates in the circuit"""
    cnot_count = 0
    ry_count = 0
    h_count = 0
    
    for instruction in circuit.data:
        name = instruction[0].name
        if 'cx' in name or 'cnot' in name:
            cnot_count += 1
        elif 'ry' in name:
            ry_count += 1
        elif name == 'h':
            h_count += 1
    
    return cnot_count, ry_count, h_count

def main():
    parser = argparse.ArgumentParser(
        description='QPIXL Qiskit: Compile quantum circuits for image representations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python frqi_qiskit.py input.pgm output 50 1
  
  This will read input.pgm, apply 50%% compression, and simulate the circuit
        """
    )
    
    parser.add_argument('input', help='Path to input PGM image file')
    parser.add_argument('output', help='Path and name structure for output files')
    parser.add_argument('compression', nargs='?', type=float, default=0.0,
                       help='Compression level (0-100 percent), default: 0')
    parser.add_argument('simulate', nargs='?', type=int, default=0,
                       help='Simulate flag (0 or 1), default: 0')
    
    args = parser.parse_args()
    
    # Read image data
    print(" * Reading image data from file..")
    try:
        data, nrows, ncols, maxval = read_pgm_ascii(args.input)
    except Exception as e:
        print(f"ERROR -- reading input image file: {e}")
        return -2
    
    # Zero padding
    data = zero_pad_to_power_of_2(data)
    
    # Image statistics
    print(f"\n   Image statistics:")
    print(f"     * number of rows             : {nrows}")
    print(f"     * number of columns          : {ncols}")
    print(f"     * maximum pixel value        : {maxval}")
    print(f"     * size of zero padded vector : {len(data)}\n")
    
    # Convert grayscale to angles
    print(" * Converting grayscale to angles..")
    angles = convert_to_angles(data, maxval)
    
    # Create compressed FRQI circuit
    print(" * Compressing FRQI circuit..")
    circuit = create_compressed_frqi_circuit(angles, args.compression)
    
    # Count gates
    n_cnot, n_ry, n_h = count_circuit_gates(circuit)
    n_qubits = circuit.num_qubits
    max_gates = 2 ** (n_qubits - 1)
    
    cnot_compression = (1.0 - (n_cnot / max_gates)) * 100
    ry_compression = (1.0 - (n_ry / max_gates)) * 100
    
    # Write to QASM file
    print(" * Writing compressed FRQI circuit to QASM..")
    qasm_file = args.output + ".qasm"
    
    with open(qasm_file, 'w') as f:
        f.write("// Generated by QPIXL Qiskit\n")
        f.write("// Converted from QPIXL++\n")
        f.write("// https://github.com/QuantumComputingLab/qpixlpp\n\n")
        f.write("//   Circuit statistics:\n")
        f.write(f"//     * number of qubits               : {n_qubits}\n")
        f.write(f"//     * total number of gates          : {len(circuit.data)}\n")
        f.write(f"//       - number of CNOTs              : {n_cnot}\n")
        f.write(f"//       - number of Ry gates           : {n_ry}\n")
        f.write(f"//       - number of Hadamard gates     : {n_h}\n")
        f.write(f"//     * Compression setting            : {args.compression}\n")
        f.write(f"//       - CNOT compression ratio [%]   : {cnot_compression:.1f}\n")
        f.write(f"//       - Ry compression ratio [%]     : {ry_compression:.1f}\n\n")
        f.write(circuit.qasm())
    
    # Print circuit statistics
    print(f"\n   Circuit statistics:")
    print(f"     * number of qubits               : {n_qubits}")
    print(f"     * total number of gates          : {len(circuit.data)}")
    print(f"       - number of CNOTs              : {n_cnot}")
    print(f"       - number of Ry gates           : {n_ry}")
    print(f"       - number of Hadamard gates     : {n_h}")
    print(f"     * Compression setting            : {args.compression}")
    print(f"       - CNOT compression ratio [%]   : {cnot_compression:.1f}")
    print(f"       - Ry compression ratio [%]     : {ry_compression:.1f}\n")
    
    # Transform data back to grayscale
    print(" * Converting compressed data to grayscale image..")
    data_back = angles.copy()
    inv_gray_permutation(data_back)
    isfwht(data_back)
    data_back /= 2
    grayscale_data = convert_to_grayscale(data_back, maxval)
    
    # Write to PGM
    print(" * Writing compressed image to file..")
    pgm_file = args.output + ".pgm"
    write_pgm_ascii(pgm_file, grayscale_data, nrows, ncols, maxval)
    
    # Simulate the circuit
    if args.simulate == 1:
        print(" * Simulating the compressed FRQI circuit..")
        
        simulator = AerSimulator()
        circuit_copy = circuit.copy()
        circuit_copy.save_statevector()
        
        result = simulator.run(circuit_copy).result()
        statevector = result.get_statevector()
        
        # Recover image data from state vector
        state_array = np.array(statevector)
        recovered_angles = []
        
        for i in range(0, len(state_array), 2):
            angle = np.arctan2(state_array[i+1].real, state_array[i].real)
            recovered_angles.append(angle)
        
        recovered_angles = np.array(recovered_angles[:len(data)])
        sim_grayscale = convert_to_grayscale(recovered_angles, maxval)
        
        # Write simulated image
        print(" * Writing simulated compressed image to file..")
        sim_file = args.output + "_sim.pgm"
        write_pgm_ascii(sim_file, sim_grayscale, nrows, ncols, maxval)
    
    print("\n Completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())