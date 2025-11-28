# QPIXL++ FRQI Implementation - Complete Testing Guide

## Overview

This repository implements **Flexible Representation of Quantum Images (FRQI)** using Qiskit, a Python quantum computing framework. FRQI encodes grayscale images into quantum circuits using compressed representations that require quadratically fewer gates than traditional methods.

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install numpy qiskit qiskit-aer
```

Optional for image conversion:
```bash
sudo apt-get install imagemagick
```

---

## Part 1: Converting Images to PGM Format

### What is PGM?

PGM (Portable Grayscale Map) is a simple ASCII format for grayscale images:

```
P2
4 4
255
255 200 150 100
200 150 100 50
150 100 50 0
100 50 0 255
```

- Line 1: `P2` = ASCII grayscale format
- Line 2: `width height` (columns rows)
- Line 3: Maximum pixel value (usually 255)
- Lines 4+: Pixel values (0-255, where 0=black, 255=white)

### Option A: Use Existing Example Images

The repository includes several test images in `/examples/`:

```bash
ls -lh examples/*.pgm
```

Available images:
- `Example0.pgm` - 600×600 pixels (1.5 MB)
- `Example1.pgm` - 512×512 pixels (1.4 MB)
- `Example2.pgm`, `Example3.pgm`, `Example4.pgm`, `Example5.pgm`

### Option B: Convert Your Own Images

Using ImageMagick:

```bash
# Basic conversion (PNG/JPG to PGM)
convert your_image.jpg -colorspace Gray -compress none output.pgm

# Resize to specific dimensions (recommended: powers of 2)
convert your_image.jpg -colorspace Gray -resize 512x512! -compress none output.pgm

# Resize to fit within dimensions (maintains aspect ratio)
convert your_image.jpg -colorspace Gray -resize 256x256 -compress none output.pgm
```

Using Python (PIL/Pillow):

```python
from PIL import Image
import numpy as np

# Load and convert to grayscale
img = Image.open('your_image.jpg').convert('L')

# Resize to power of 2 (optional but recommended)
img = img.resize((512, 512))

# Save as PGM
img.save('output.pgm')
```

### Option C: Create Test Images Programmatically

```python
import numpy as np

def create_test_pgm(filename, size=8):
    """Create a simple gradient test image"""
    # Create gradient pattern
    data = np.linspace(0, 255, size*size).reshape(size, size)

    with open(filename, 'w') as f:
        f.write('P2\\n')
        f.write(f'{size} {size}\\n')
        f.write('255\\n')
        for row in data:
            f.write(' '.join(map(str, row.astype(int))) + '\\n')

create_test_pgm('gradient.pgm', size=16)
```

---

## Part 2: Running the FRQI Implementation

### Method 1: Main Script (compressedFRQI.py)

The primary tool for FRQI circuit generation:

```bash
cd examples
python compressedFRQI.py <input.pgm> <output> <compression> <simulate>
```

**Parameters:**
- `input.pgm` - Path to input PGM image file
- `output` - Output filename prefix (without extension)
- `compression` - Compression level (0-100%)
  - `0` = No compression (all coefficients encoded)
  - `50` = 50% of smallest coefficients zeroed
  - `100` = Full compression (no coefficients)
- `simulate` - Run circuit simulation
  - `0` = No simulation (faster)
  - `1` = Run simulation (slower, requires more memory)

**Examples:**

```bash
# No compression, no simulation
python compressedFRQI.py Example4.pgm output_no_comp 0 0

# 50% compression, no simulation
python compressedFRQI.py Example4.pgm output_50 50 0

# 90% compression, no simulation (very sparse circuit)
python compressedFRQI.py Example4.pgm output_90 90 0

# 25% compression with simulation
python compressedFRQI.py Example4.pgm output_sim 25 1
```

**Outputs:**
- `output.qasm` - OpenQASM 2.0 quantum circuit file
- `output.pgm` - Compressed/reconstructed PGM image
- `output_sim.pgm` - Simulated result (if simulation enabled)

### Method 2: Python API (Programmatic Usage)

Basic example using the FRQI circuit directly:

```python
import numpy as np
from include.qpixl.frqi.circuit_qiskit import compressed_frqi_circuit

# Create angle representation (4 pixels, normalized to [0, π/2])
angles = np.array([0.5, 1.0, 1.5, 2.0])

# Generate circuit with no compression
qc = compressed_frqi_circuit(angles.copy(), compression=0.0)

print(qc)
print(f"Qubits: {qc.num_qubits}")
print(f"Depth: {qc.depth()}")
print(f"Gates: {qc.size()}")
```

Complete example with image loading:

```python
import numpy as np
from include.qpixl.pgm import read_pgma, write_pgma
from include.qpixl.frqi.util import convert_to_angles
from include.qpixl.frqi.circuit_qiskit import compressed_frqi_circuit

# 1. Load PGM image
data, nrows, ncols, maxval = read_pgma('examples/Example4.pgm', zero_pad=2)

# 2. Convert to angles
angles = convert_to_angles(data, maxval)

# 3. Generate compressed FRQI circuit
qc = compressed_frqi_circuit(angles, compression=50.0)

# 4. Print circuit statistics
print(f"Image: {nrows}x{ncols}, Max value: {maxval}")
print(f"Quantum Circuit:")
print(f"  - Qubits: {qc.num_qubits}")
print(f"  - Depth: {qc.depth()}")
print(f"  - Total gates: {qc.size()}")

# 5. Export to QASM
from qiskit.qasm2 import dumps
qasm_str = dumps(qc)
with open('output.qasm', 'w') as f:
    f.write(qasm_str)
```

### Method 3: Interactive Testing

Run the provided example script:

```bash
python example_usage.py
```

This demonstrates:
- 4-pixel example with and without compression
- 8-pixel example
- Circuit simulation with statevector

---

## Part 3: Understanding FRQI Implementation

### How FRQI Works

**1. Image Encoding:**
   - Grayscale image → 1D vector of pixel values
   - Normalize: pixels (0-255) → angles (0 to π/2)
   - Formula: `angle = (pixel / max_value) * (π/2)`

**2. Quantum Representation:**
   ```
   |I⟩ = 1/√N ∑(i=0 to N-1) (cos θᵢ|0⟩ + sin θᵢ|1⟩) ⊗ |i⟩
   ```
   - `N` = number of pixels (must be power of 2)
   - `θᵢ` = angle for pixel i
   - `|i⟩` = binary encoding of pixel position
   - `|0⟩/|1⟩` = color qubit (intensity)

**3. Compression Transform:**
   - Apply **Scaled Fast Walsh-Hadamard Transform (SFWHT)**
   - Apply **Gray code permutation**
   - Zero smallest coefficients (compression)
   - Generates sparse circuit

**4. Circuit Structure:**
   - **k qubits** for position encoding (k = log₂(N))
   - **1 qubit** for color/intensity
   - **Total: k+1 qubits**

   Example: 256-pixel image needs 8+1=9 qubits

**5. Gates Used:**
   - **Hadamard (H)** - Create superposition on position qubits
   - **Controlled-RY** - Encode pixel intensities
   - **CNOT (CX)** - Implement control logic
   - NISQ-friendly (no multi-qubit gates beyond CNOT)

### Key Functions Explained

**From `/include/qpixl/frqi/circuit_qiskit.py`:**

```python
def compressed_frqi_circuit(a, compression=0.0):
    """
    Generate compressed FRQI quantum circuit

    Args:
        a: Angle array (size must be power of 2)
        compression: 0-100, percentage of smallest coefficients to zero

    Returns:
        QuantumCircuit: Compressed FRQI circuit
    """
```

**From `/include/qpixl/frqi/util.py`:**

- `sfwht(a)` - Scaled Fast Walsh-Hadamard Transform
- `gray_permutation(a)` - Apply Gray code reordering
- `convert_to_angles(data, maxval)` - Grayscale to angles
- `convert_to_grayscale(angles, maxval)` - Angles to grayscale

**From `/include/qpixl/pgm.py`:**

- `read_pgma(filename, zero_pad=0)` - Read PGM files
  - `zero_pad=0`: No padding
  - `zero_pad=1`: Embed in 2^n × 2^m image
  - `zero_pad=2`: Pad vector to power of 2
- `write_pgma(filename, data, nrows, ncols, maxval)` - Write PGM files

### Circuit Optimization

**Compression Benefits:**
- 0% compression: Full fidelity, O(N) gates
- 50% compression: 2× fewer gates, minimal quality loss
- 90% compression: 10× fewer gates, noticeable quality loss
- 99% compression: 100× fewer gates, significant degradation

**Gate Count Reduction:**
```
Uncompressed: ~2^k RY gates + CNOTs
Compressed (50%): ~2^(k-1) RY gates
```

### Example Output Analysis

Running `python compressedFRQI.py Example4.pgm test 50 0`:

```
Image statistics:
  * number of rows             : 28
  * number of columns          : 28
  * maximum pixel value        : 255
  * size of zero padded vector : 1024

Circuit statistics:
  * number of qubits               : 11
  * total number of gates          : 5754
    - number of CNOTs              : 0
    - number of Ry gates           : 512
    - number of Hadamard gates     : 10
  * Compression setting            : 50.0
    - CNOT compression ratio [%]   : 100.0
    - Ry compression ratio [%]     : 50.0
```

**Interpretation:**
- 28×28 = 784 pixels → padded to 1024 (2^10)
- 11 qubits needed: 10 position + 1 color
- 50% compression: 512 RY gates (down from 1024)
- Efficient implementation: minimal CNOTs

---

## Part 4: Analyzing Results

### Viewing QASM Output

The `.qasm` file contains OpenQASM 2.0 quantum assembly:

```bash
head -30 test_output.qasm
```

Example output:
```qasm
// Generated by QPIXL Qiskit
OPENQASM 2.0;
include "qelib1.inc";

// Custom gate definitions
gate c10ry(param0) q0,q1,q2,...,q10 {
  // Multi-controlled RY implementation
  h q10;
  cx q7,q10;
  ...
}

// Circuit application
qreg q[11];
h q[1];
h q[2];
...
c10ry(-2.5) q[1],q[2],...,q[0];
```

### Viewing Compressed Images

Compare original and compressed:

```bash
# View file info
file examples/Example4.pgm
file test_output.pgm

# View first few lines
head -10 examples/Example4.pgm
head -10 test_output.pgm
```

### Circuit Visualization

```python
from qiskit import QuantumCircuit
from include.qpixl.frqi.circuit_qiskit import compressed_frqi_circuit
import numpy as np

# Create small circuit for visualization
angles = np.array([0.5, 1.0, 1.5, 2.0])
qc = compressed_frqi_circuit(angles, compression=0)

# Draw circuit
print(qc.draw(output='text'))

# Get circuit properties
print(f"\\nCircuit Properties:")
print(f"  Width (qubits): {qc.num_qubits}")
print(f"  Depth (layers): {qc.depth()}")
print(f"  Size (gates): {qc.size()}")
print(f"  Operations: {qc.count_ops()}")
```

---

## Part 5: Advanced Usage

### Batch Processing Multiple Images

```python
import os
from glob import glob

input_dir = 'examples'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

for pgm_file in glob(f'{input_dir}/*.pgm'):
    basename = os.path.basename(pgm_file).replace('.pgm', '')

    for compression in [0, 25, 50, 75, 90]:
        output = f'{output_dir}/{basename}_c{compression}'
        cmd = f'python examples/compressedFRQI.py {pgm_file} {output} {compression} 0'
        print(f"Processing: {basename} @ {compression}% compression")
        os.system(cmd)
```

### Custom Circuit Analysis

```python
from qiskit.quantum_info import Statevector
import numpy as np

# Generate circuit
angles = np.array([0.5, 1.0, 1.5, 2.0])
qc = compressed_frqi_circuit(angles, compression=0)

# Get statevector
sv = Statevector(qc)

# Analyze amplitudes
print("Statevector amplitudes:")
for i, amp in enumerate(sv.data):
    if abs(amp) > 1e-10:
        print(f"  |{i:03b}⟩: {amp:.6f}")

# Verify normalization
norm = sum(abs(a)**2 for a in sv.data)
print(f"\\nNormalization: {norm:.10f} (should be 1.0)")
```

### Performance Benchmarking

```python
import time
import numpy as np

sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
compressions = [0, 50, 90]

print("Size | Comp | Qubits | Gates | Time (s)")
print("-" * 50)

for size in sizes:
    angles = np.random.rand(size) * np.pi/2

    for comp in compressions:
        start = time.time()
        qc = compressed_frqi_circuit(angles.copy(), compression=comp)
        elapsed = time.time() - start

        print(f"{size:4d} | {comp:3d}% | {qc.num_qubits:6d} | {qc.size():5d} | {elapsed:.4f}")
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'numpy'**
```bash
pip install numpy qiskit qiskit-aer
```

**2. Input size must be power of 2**
- Solution: Use `zero_pad=2` parameter when reading PGM
- Or manually pad images before processing

**3. Memory errors with large images**
- Reduce image size
- Use higher compression (75-90%)
- Disable simulation

**4. Simulation fails with "unknown instruction"**
- This is expected for large multi-controlled gates
- Run without simulation (set simulate=0)
- The circuit is still correctly generated

**5. QASM export AttributeError**
- Updated Qiskit uses `qiskit.qasm2.dumps()` instead of `circuit.qasm()`
- The fix has been applied in the current version

---

## Testing Functionality

### Quick Test Suite

```bash
# Test 1: Basic circuit generation
python example_usage.py

# Test 2: Small image processing
cd examples
python compressedFRQI.py Example4.pgm test_small 0 0

# Test 3: Compression comparison
python compressedFRQI.py Example4.pgm test_0 0 0
python compressedFRQI.py Example4.pgm test_50 50 0
python compressedFRQI.py Example4.pgm test_90 90 0

# Compare gate counts
grep "number of Ry gates" test_*.qasm
```

### Verification Checklist

- [ ] Dependencies installed (numpy, qiskit, qiskit-aer)
- [ ] Can read PGM files
- [ ] Can generate FRQI circuits
- [ ] Output QASM files created
- [ ] Compression reduces gate count
- [ ] Higher compression = fewer gates
- [ ] Example scripts run without errors

---

## References

### Papers

- [Quantum pixel representations and compression for N-dimensional images](https://arxiv.org/abs/2110.04405)
  Mercy Amankwah, Daan Camps, E. Wes Bethel, Roel Van Beeumen, Talita Perciano (2021)

### Documentation

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [OpenQASM 2.0 Specification](https://github.com/Qiskit/openqasm/tree/OpenQASM2.x)
- [Original QPIXL++ Repository](https://github.com/QuantumComputingLab/qpixlpp)

### Key Concepts

- **FRQI**: Flexible Representation of Quantum Images
- **NISQ**: Noisy Intermediate-Scale Quantum devices
- **SFWHT**: Scaled Fast Walsh-Hadamard Transform
- **Gray Code**: Binary encoding minimizing bit changes
- **Qubit**: Quantum bit (superposition of |0⟩ and |1⟩)

---

## Example Workflow

Complete end-to-end example:

```bash
# 1. Create test image
python -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 256, (16,16), dtype=np.uint8))
img.save('my_test.pgm')
"

# 2. Convert to PGM (if needed)
convert my_image.jpg -colorspace Gray -resize 256x256! -compress none my_image.pgm

# 3. Generate FRQI circuit with 50% compression
cd examples
python compressedFRQI.py my_image.pgm output 50 0

# 4. View results
head -30 output.qasm  # View circuit
file output.pgm       # Verify output image

# 5. Compare different compressions
for comp in 0 25 50 75 90 95 99; do
    python compressedFRQI.py my_image.pgm out_$comp $comp 0
    grep "Ry gates" out_$comp.qasm
done
```

---

## Summary

**FRQI Quantum Image Encoding:**
- Converts grayscale images to quantum circuits
- Uses k+1 qubits for N=2^k pixels
- Compression reduces circuit size quadratically
- NISQ-friendly (only CNOTs and RY gates)
- Useful for quantum image processing research

**Typical Use Cases:**
- Quantum machine learning with image data
- Quantum image compression studies
- NISQ algorithm development
- Quantum circuit optimization research

**Performance:**
- Small images (16×16): Near-instant
- Medium images (256×256): Seconds
- Large images (512×512): Minutes
- Very large (1024×1024): Consider using compression

---

For questions or issues, refer to:
- GitHub: https://github.com/QuantumComputingLab/qpixlpp
- Qiskit Slack: https://qiskit.slack.com
