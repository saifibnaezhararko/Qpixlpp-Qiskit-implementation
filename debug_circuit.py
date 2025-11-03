import numpy as np
import sys
sys.path.insert(0, r'e:\qpixlpp-master\include')
from include.qpixl.frqi.circuit_qiskit import compressed_frqi_circuit

b = np.array([0.5, 1.0, 1.5, 2.0])
print("Input:", b)
qc = compressed_frqi_circuit(b.copy())
print(qc)
print('\nGate sequence:')
for i, gate in enumerate(qc.data):
    qubits_str = ', '.join([f'q[{q._index}]' for q in gate.qubits])
    if gate.operation.name == 'ry':
        print(f'{i}: {gate.operation.name}({gate.operation.params[0]:.4f}) on {qubits_str}')
    else:
        print(f'{i}: {gate.operation.name} on {qubits_str}')
