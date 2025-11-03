"""
Example: Using the QPIXL FRQI Circuit Generator

This example demonstrates how to use the compressed FRQI circuit
generation for quantum image processing.
"""

import numpy as np
import sys
sys.path.insert(0, r'e:\qpixlpp-master\include')
from include.qpixl.frqi.circuit_qiskit import compressed_frqi_circuit

# Example 1: Simple 4-pixel image
print("=" * 60)
print("Example 1: 4-pixel Image (2x2)")
print("=" * 60)

# Create angle representation (normalized pixel values)
angles = np.array([0.5, 1.0, 1.5, 2.0])
print(f"Input angles: {angles}")

# Generate FRQI circuit with no compression
qc = compressed_frqi_circuit(angles.copy(), compression=0.0)
print(f"\nCircuit (no compression):")
print(qc)
print(f"  - Number of qubits: {qc.num_qubits}")
print(f"  - Circuit depth: {qc.depth()}")
print(f"  - Total gates: {qc.size()}")

# Generate FRQI circuit with 25% compression
qc_compressed = compressed_frqi_circuit(angles.copy(), compression=25.0)
print(f"\nCircuit (25% compression):")
print(qc_compressed)
print(f"  - Number of qubits: {qc_compressed.num_qubits}")
print(f"  - Circuit depth: {qc_compressed.depth()}")
print(f"  - Total gates: {qc_compressed.size()}")

# Example 2: Larger image (8 pixels)
print("\n" + "=" * 60)
print("Example 2: 8-pixel Image")
print("=" * 60)

angles_8 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
print(f"Input angles: {angles_8}")

qc_8 = compressed_frqi_circuit(angles_8.copy(), compression=0.0)
print(f"\nCircuit:")
print(qc_8)
print(f"  - Number of qubits: {qc_8.num_qubits}")
print(f"  - Circuit depth: {qc_8.depth()}")
print(f"  - Total gates: {qc_8.size()}")

# Example 3: Simulate the circuit
print("\n" + "=" * 60)
print("Example 3: Circuit Simulation")
print("=" * 60)

from qiskit.quantum_info import Statevector

# Use a simple 2-pixel example for clear output
angles_2 = np.array([0.5, 1.0])
qc_2 = compressed_frqi_circuit(angles_2.copy(), compression=0.0)

# Get the statevector
statevector = Statevector(qc_2)
print(f"Statevector amplitudes:")
for i, amp in enumerate(statevector.data):
    if abs(amp) > 1e-10:  # Only show non-zero amplitudes
        print(f"  |{i:02b}⟩: {amp:.6f}")

print("\n" + "=" * 60)
print("✅ QPIXL library is successfully installed and working!")
print("=" * 60)
