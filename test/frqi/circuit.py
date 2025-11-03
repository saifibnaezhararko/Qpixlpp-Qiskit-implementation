import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import sys
import os

# Add parent directory to path to import from include folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'include'))
from include.qpixl.frqi.circuit_qiskit import compressed_frqi_circuit


def check_matrix(m1, m2, eps):
    """
    Compares two matrices
    
    Args:
        m1: first matrix
        m2: second matrix
        eps: tolerance value
    """
    assert m1.shape == m2.shape, f"Matrix shapes don't match: {m1.shape} vs {m2.shape}"
    
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            assert np.abs(m1[i, j] - m2[i, j]) < eps, \
                f"Values don't match at position ({i},{j}): {m1[i,j]} vs {m2[i,j]}"


def test_compressed_frqi_circuit_case_a():
    """Test Case A: 2 elements"""
    eps = np.finfo(np.float64).eps
    
    a = np.array([0.5, 1.0])
    
    # Create circuit (pass a copy since function modifies array in-place)
    qc = compressed_frqi_circuit(a.copy())
    
    # Get unitary matrix of the circuit
    operator = Operator(qc)
    circ_a_matrix = operator.data
    
    # Expected matrix
    circ_a_check = np.array([
        [0.620544580563746, -0.339005049421045,  0.620544580563746, -0.339005049421045],
        [0.339005049421045,  0.620544580563746,  0.339005049421045,  0.620544580563746],
        [0.382051424370090, -0.595009839529386, -0.382051424370090,  0.595009839529386],
        [0.595009839529386,  0.382051424370090, -0.595009839529386, -0.382051424370090]
    ])
    
    check_matrix(circ_a_matrix, circ_a_check, 10 * eps)
    print("Test A passed!")


def test_compressed_frqi_circuit_case_b():
    """Test Case B: 4 elements"""
    eps = np.finfo(np.float64).eps
    
    b = np.array([0.5, 1.0, 1.5, 2.0])
    
    # Create circuit (pass a copy since function modifies array in-place)
    qc = compressed_frqi_circuit(b.copy())
    
    # Get unitary matrix of the circuit
    operator = Operator(qc)
    circ_b_matrix = operator.data
    
    # Expected matrix (8x8)
    circ_b_check = np.zeros((8, 8))
    
    # Column 0
    circ_b_check[0, 0] =  0.438791280945186
    circ_b_check[1, 0] =  0.239712769302101
    circ_b_check[2, 0] =  0.270151152934070
    circ_b_check[3, 0] =  0.420735492403948
    circ_b_check[4, 0] =  0.035368600833851
    circ_b_check[5, 0] =  0.498747493302027
    circ_b_check[6, 0] = -0.208073418273571
    circ_b_check[7, 0] =  0.454648713412841
    
    # Column 1
    circ_b_check[0, 1] = -0.239712769302101
    circ_b_check[1, 1] =  0.438791280945186
    circ_b_check[2, 1] = -0.420735492403948
    circ_b_check[3, 1] =  0.270151152934070
    circ_b_check[4, 1] = -0.498747493302027
    circ_b_check[5, 1] =  0.035368600833851
    circ_b_check[6, 1] = -0.454648713412841
    circ_b_check[7, 1] = -0.208073418273571
    
    # Column 2
    circ_b_check[0, 2] =  0.438791280945186
    circ_b_check[1, 2] =  0.239712769302101
    circ_b_check[2, 2] = -0.270151152934070
    circ_b_check[3, 2] = -0.420735492403948
    circ_b_check[4, 2] =  0.035368600833851
    circ_b_check[5, 2] =  0.498747493302027
    circ_b_check[6, 2] =  0.208073418273571
    circ_b_check[7, 2] = -0.454648713412841
    
    # Column 3
    circ_b_check[0, 3] = -0.239712769302101
    circ_b_check[1, 3] =  0.438791280945186
    circ_b_check[2, 3] =  0.420735492403948
    circ_b_check[3, 3] = -0.270151152934070
    circ_b_check[4, 3] = -0.498747493302027
    circ_b_check[5, 3] =  0.035368600833851
    circ_b_check[6, 3] =  0.454648713412841
    circ_b_check[7, 3] =  0.208073418273571
    
    # Column 4
    circ_b_check[0, 4] =  0.438791280945186
    circ_b_check[1, 4] =  0.239712769302101
    circ_b_check[2, 4] =  0.270151152934070
    circ_b_check[3, 4] =  0.420735492403948
    circ_b_check[4, 4] = -0.035368600833851
    circ_b_check[5, 4] = -0.498747493302027
    circ_b_check[6, 4] =  0.208073418273571
    circ_b_check[7, 4] = -0.454648713412841
    
    # Column 5
    circ_b_check[0, 5] = -0.239712769302101
    circ_b_check[1, 5] =  0.438791280945186
    circ_b_check[2, 5] = -0.420735492403948
    circ_b_check[3, 5] =  0.270151152934070
    circ_b_check[4, 5] =  0.498747493302027
    circ_b_check[5, 5] = -0.035368600833851
    circ_b_check[6, 5] =  0.454648713412841
    circ_b_check[7, 5] =  0.208073418273571
    
    # Column 6
    circ_b_check[0, 6] =  0.438791280945186
    circ_b_check[1, 6] =  0.239712769302101
    circ_b_check[2, 6] = -0.270151152934070
    circ_b_check[3, 6] = -0.420735492403948
    circ_b_check[4, 6] = -0.035368600833851
    circ_b_check[5, 6] = -0.498747493302027
    circ_b_check[6, 6] = -0.208073418273571
    circ_b_check[7, 6] =  0.454648713412841
    
    # Column 7
    circ_b_check[0, 7] = -0.239712769302101
    circ_b_check[1, 7] =  0.438791280945186
    circ_b_check[2, 7] =  0.420735492403948
    circ_b_check[3, 7] = -0.270151152934070
    circ_b_check[4, 7] =  0.498747493302027
    circ_b_check[5, 7] = -0.035368600833851
    circ_b_check[6, 7] = -0.454648713412841
    circ_b_check[7, 7] = -0.208073418273571
    
    check_matrix(circ_b_matrix, circ_b_check, 10 * eps)
    print("Test B passed!")


def test_compressed_frqi_circuit_case_c():
    """Test Case C: 8 elements"""
    eps = np.finfo(np.float64).eps
    
    c = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    
    # Create circuit (pass a copy since function modifies array in-place)
    qc = compressed_frqi_circuit(c.copy())
    
    # Get unitary matrix of the circuit
    operator = Operator(qc)
    circ_c_matrix = operator.data
    
    # Expected matrix (16x16)
    circ_c_check = np.zeros((16, 16))
    
    # Column 0
    circ_c_check[0, 0] = 0.310272290281873
    circ_c_check[1, 0] = 0.169502524710522
    circ_c_check[2, 0] = 0.191025712185045
    circ_c_check[3, 0] = 0.297504919764693
    circ_c_check[4, 0] = 0.025009377490697
    circ_c_check[5, 0] = 0.352667734613656
    circ_c_check[6, 0] = -0.147130125045907
    circ_c_check[7, 0] = 0.321485188311959
    circ_c_check[8, 0] = -0.283247041628773
    circ_c_check[9, 0] = 0.211591855723580
    circ_c_check[10, 0] = -0.350015203834987
    circ_c_check[11, 0] = 0.049893457330116
    circ_c_check[12, 0] = -0.331087436935406
    circ_c_check[13, 0] = -0.124020599512917
    circ_c_check[14, 0] = -0.231097918395994
    circ_c_check[15, 0] = -0.267570088225568
    
    # Column 1
    circ_c_check[0, 1] = -0.169502524710522
    circ_c_check[1, 1] = 0.310272290281873
    circ_c_check[2, 1] = -0.297504919764693
    circ_c_check[3, 1] = 0.191025712185045
    circ_c_check[4, 1] = -0.352667734613656
    circ_c_check[5, 1] = 0.025009377490697
    circ_c_check[6, 1] = -0.321485188311959
    circ_c_check[7, 1] = -0.147130125045907
    circ_c_check[8, 1] = -0.211591855723580
    circ_c_check[9, 1] = -0.283247041628773
    circ_c_check[10, 1] = -0.049893457330116
    circ_c_check[11, 1] = -0.350015203834987
    circ_c_check[12, 1] = 0.124020599512917
    circ_c_check[13, 1] = -0.331087436935406
    circ_c_check[14, 1] = 0.267570088225568
    circ_c_check[15, 1] = -0.231097918395994
    
    # Column 2
    circ_c_check[0, 2] = 0.310272290281873
    circ_c_check[1, 2] = 0.169502524710522
    circ_c_check[2, 2] = -0.191025712185045
    circ_c_check[3, 2] = -0.297504919764693
    circ_c_check[4, 2] = 0.025009377490697
    circ_c_check[5, 2] = 0.352667734613656
    circ_c_check[6, 2] = 0.147130125045907
    circ_c_check[7, 2] = -0.321485188311959
    circ_c_check[8, 2] = -0.283247041628773
    circ_c_check[9, 2] = 0.211591855723580
    circ_c_check[10, 2] = 0.350015203834987
    circ_c_check[11, 2] = -0.049893457330116
    circ_c_check[12, 2] = -0.331087436935406
    circ_c_check[13, 2] = -0.124020599512917
    circ_c_check[14, 2] = 0.231097918395994
    circ_c_check[15, 2] = 0.267570088225568
    
    # Column 3
    circ_c_check[0, 3] = -0.169502524710522
    circ_c_check[1, 3] = 0.310272290281873
    circ_c_check[2, 3] = 0.297504919764693
    circ_c_check[3, 3] = -0.191025712185045
    circ_c_check[4, 3] = -0.352667734613656
    circ_c_check[5, 3] = 0.025009377490697
    circ_c_check[6, 3] = 0.321485188311959
    circ_c_check[7, 3] = 0.147130125045907
    circ_c_check[8, 3] = -0.211591855723580
    circ_c_check[9, 3] = -0.283247041628773
    circ_c_check[10, 3] = 0.049893457330116
    circ_c_check[11, 3] = 0.350015203834987
    circ_c_check[12, 3] = 0.124020599512917
    circ_c_check[13, 3] = -0.331087436935406
    circ_c_check[14, 3] = -0.267570088225568
    circ_c_check[15, 3] = 0.231097918395994
    
    # Column 4
    circ_c_check[0, 4] = 0.310272290281873
    circ_c_check[1, 4] = 0.169502524710522
    circ_c_check[2, 4] = 0.191025712185045
    circ_c_check[3, 4] = 0.297504919764693
    circ_c_check[4, 4] = -0.025009377490697
    circ_c_check[5, 4] = -0.352667734613656
    circ_c_check[6, 4] = 0.147130125045907
    circ_c_check[7, 4] = -0.321485188311959
    circ_c_check[8, 4] = -0.283247041628773
    circ_c_check[9, 4] = 0.211591855723580
    circ_c_check[10, 4] = -0.350015203834987
    circ_c_check[11, 4] = 0.049893457330116
    circ_c_check[12, 4] = 0.331087436935406
    circ_c_check[13, 4] = 0.124020599512917
    circ_c_check[14, 4] = 0.231097918395994
    circ_c_check[15, 4] = 0.267570088225568
    
    # Column 5
    circ_c_check[0, 5] = -0.169502524710522
    circ_c_check[1, 5] = 0.310272290281873
    circ_c_check[2, 5] = -0.297504919764693
    circ_c_check[3, 5] = 0.191025712185045
    circ_c_check[4, 5] = 0.352667734613656
    circ_c_check[5, 5] = -0.025009377490697
    circ_c_check[6, 5] = 0.321485188311959
    circ_c_check[7, 5] = 0.147130125045907
    circ_c_check[8, 5] = -0.211591855723580
    circ_c_check[9, 5] = -0.283247041628773
    circ_c_check[10, 5] = -0.049893457330116
    circ_c_check[11, 5] = -0.350015203834987
    circ_c_check[12, 5] = -0.124020599512917
    circ_c_check[13, 5] = 0.331087436935406
    circ_c_check[14, 5] = -0.267570088225568
    circ_c_check[15, 5] = 0.231097918395994
    
    # Column 6
    circ_c_check[0, 6] = 0.310272290281873
    circ_c_check[1, 6] = 0.169502524710522
    circ_c_check[2, 6] = -0.191025712185045
    circ_c_check[3, 6] = -0.297504919764693
    circ_c_check[4, 6] = -0.025009377490697
    circ_c_check[5, 6] = -0.352667734613656
    circ_c_check[6, 6] = -0.147130125045907
    circ_c_check[7, 6] = 0.321485188311959
    circ_c_check[8, 6] = -0.283247041628773
    circ_c_check[9, 6] = 0.211591855723580
    circ_c_check[10, 6] = 0.350015203834987
    circ_c_check[11, 6] = -0.049893457330116
    circ_c_check[12, 6] = 0.331087436935406
    circ_c_check[13, 6] = 0.124020599512917
    circ_c_check[14, 6] = -0.231097918395994
    circ_c_check[15, 6] = -0.267570088225568
    
    # Column 7
    circ_c_check[0, 7] = -0.169502524710522
    circ_c_check[1, 7] = 0.310272290281873
    circ_c_check[2, 7] = 0.297504919764693
    circ_c_check[3, 7] = -0.191025712185045
    circ_c_check[4, 7] = 0.352667734613656
    circ_c_check[5, 7] = -0.025009377490697
    circ_c_check[6, 7] = -0.321485188311959
    circ_c_check[7, 7] = -0.147130125045907
    circ_c_check[8, 7] = -0.211591855723580
    circ_c_check[9, 7] = -0.283247041628773
    circ_c_check[10, 7] = 0.049893457330116
    circ_c_check[11, 7] = 0.350015203834987
    circ_c_check[12, 7] = -0.124020599512917
    circ_c_check[13, 7] = 0.331087436935406
    circ_c_check[14, 7] = 0.267570088225568
    circ_c_check[15, 7] = -0.231097918395994
    
    # Column 8
    circ_c_check[0, 8] = 0.310272290281873
    circ_c_check[1, 8] = 0.169502524710522
    circ_c_check[2, 8] = 0.191025712185045
    circ_c_check[3, 8] = 0.297504919764693
    circ_c_check[4, 8] = 0.025009377490697
    circ_c_check[5, 8] = 0.352667734613656
    circ_c_check[6, 8] = -0.147130125045907
    circ_c_check[7, 8] = 0.321485188311959
    circ_c_check[8, 8] = 0.283247041628773
    circ_c_check[9, 8] = -0.211591855723580
    circ_c_check[10, 8] = 0.350015203834987
    circ_c_check[11, 8] = -0.049893457330116
    circ_c_check[12, 8] = 0.331087436935406
    circ_c_check[13, 8] = 0.124020599512917
    circ_c_check[14, 8] = 0.231097918395994
    circ_c_check[15, 8] = 0.267570088225568
    
    # Column 9
    circ_c_check[0, 9] = -0.169502524710522
    circ_c_check[1, 9] = 0.310272290281873
    circ_c_check[2, 9] = -0.297504919764693
    circ_c_check[3, 9] = 0.191025712185045
    circ_c_check[4, 9] = -0.352667734613656
    circ_c_check[5, 9] = 0.025009377490697
    circ_c_check[6, 9] = -0.321485188311959
    circ_c_check[7, 9] = -0.147130125045907
    circ_c_check[8, 9] = 0.211591855723580
    circ_c_check[9, 9] = 0.283247041628773
    circ_c_check[10, 9] = 0.049893457330116
    circ_c_check[11, 9] = 0.350015203834987
    circ_c_check[12, 9] = -0.124020599512917
    circ_c_check[13, 9] = 0.331087436935406
    circ_c_check[14, 9] = -0.267570088225568
    circ_c_check[15, 9] = 0.231097918395994
    
    # Column 10
    circ_c_check[0, 10] = 0.310272290281873
    circ_c_check[1, 10] = 0.169502524710522
    circ_c_check[2, 10] = -0.191025712185045
    circ_c_check[3, 10] = -0.297504919764693
    circ_c_check[4, 10] = 0.025009377490697
    circ_c_check[5, 10] = 0.352667734613656
    circ_c_check[6, 10] = 0.147130125045907
    circ_c_check[7, 10] = -0.321485188311959
    circ_c_check[8, 10] = 0.283247041628773
    circ_c_check[9, 10] = -0.211591855723580
    circ_c_check[10, 10] = -0.350015203834987
    circ_c_check[11, 10] = 0.049893457330116
    circ_c_check[12, 10] = 0.331087436935406
    circ_c_check[13, 10] = 0.124020599512917
    circ_c_check[14, 10] = -0.231097918395994
    circ_c_check[15, 10] = -0.267570088225568
    
    # Column 11
    circ_c_check[0, 11] = -0.169502524710522
    circ_c_check[1, 11] = 0.310272290281873
    circ_c_check[2, 11] = 0.297504919764693
    circ_c_check[3, 11] = -0.191025712185045
    circ_c_check[4, 11] = -0.352667734613656
    circ_c_check[5, 11] = 0.025009377490697
    circ_c_check[6, 11] = 0.321485188311959
    circ_c_check[7, 11] = 0.147130125045907
    circ_c_check[8, 11] = 0.211591855723580
    circ_c_check[9, 11] = 0.283247041628773
    circ_c_check[10, 11] = -0.049893457330116
    circ_c_check[11, 11] = -0.350015203834987
    circ_c_check[12, 11] = -0.124020599512917
    circ_c_check[13, 11] = 0.331087436935406
    circ_c_check[14, 11] = 0.267570088225568
    circ_c_check[15, 11] = -0.231097918395994
    
    # Column 12
    circ_c_check[0, 12] = 0.310272290281873
    circ_c_check[1, 12] = 0.169502524710522
    circ_c_check[2, 12] = 0.191025712185045
    circ_c_check[3, 12] = 0.297504919764693
    circ_c_check[4, 12] = -0.025009377490697
    circ_c_check[5, 12] = -0.352667734613656
    circ_c_check[6, 12] = 0.147130125045907
    circ_c_check[7, 12] = -0.321485188311959
    circ_c_check[8, 12] = 0.283247041628773
    circ_c_check[9, 12] = -0.211591855723580
    circ_c_check[10, 12] = 0.350015203834987
    circ_c_check[11, 12] = -0.049893457330116
    circ_c_check[12, 12] = -0.331087436935406
    circ_c_check[13, 12] = -0.124020599512917
    circ_c_check[14, 12] = -0.231097918395994
    circ_c_check[15, 12] = -0.267570088225568
    
    # Column 13
    circ_c_check[0, 13] = -0.169502524710522
    circ_c_check[1, 13] = 0.310272290281873
    circ_c_check[2, 13] = -0.297504919764693
    circ_c_check[3, 13] = 0.191025712185045
    circ_c_check[4, 13] = 0.352667734613656
    circ_c_check[5, 13] = -0.025009377490697
    circ_c_check[6, 13] = 0.321485188311959
    circ_c_check[7, 13] = 0.147130125045907
    circ_c_check[8, 13] = 0.211591855723580
    circ_c_check[9, 13] = 0.283247041628773
    circ_c_check[10, 13] = 0.049893457330116
    circ_c_check[11, 13] = 0.350015203834987
    circ_c_check[12, 13] = 0.124020599512917
    circ_c_check[13, 13] = -0.331087436935406
    circ_c_check[14, 13] = 0.267570088225568
    circ_c_check[15, 13] = -0.231097918395994
    
    # Column 14
    circ_c_check[0, 14] = 0.310272290281873
    circ_c_check[1, 14] = 0.169502524710522
    circ_c_check[2, 14] = -0.191025712185045
    circ_c_check[3, 14] = -0.297504919764693
    circ_c_check[4, 14] = -0.025009377490697
    circ_c_check[5, 14] = -0.352667734613656
    circ_c_check[6, 14] = -0.147130125045907
    circ_c_check[7, 14] = 0.321485188311959
    circ_c_check[8, 14] = 0.283247041628773
    circ_c_check[9, 14] = -0.211591855723580
    circ_c_check[10, 14] = -0.350015203834987
    circ_c_check[11, 14] = 0.049893457330116
    circ_c_check[12, 14] = -0.331087436935406
    circ_c_check[13, 14] = -0.124020599512917
    circ_c_check[14, 14] = 0.231097918395994
    circ_c_check[15, 14] = 0.267570088225568
    
    # Column 15
    circ_c_check[0, 15] = -0.169502524710522
    circ_c_check[1, 15] = 0.310272290281873
    circ_c_check[2, 15] = 0.297504919764693
    circ_c_check[3, 15] = -0.191025712185045
    circ_c_check[4, 15] = 0.352667734613656
    circ_c_check[5, 15] = -0.025009377490697
    circ_c_check[6, 15] = -0.321485188311959
    circ_c_check[7, 15] = -0.147130125045907
    circ_c_check[8, 15] = 0.211591855723580
    circ_c_check[9, 15] = 0.283247041628773
    circ_c_check[10, 15] = -0.049893457330116
    circ_c_check[11, 15] = -0.350015203834987
    circ_c_check[12, 15] = 0.124020599512917
    circ_c_check[13, 15] = -0.331087436935406
    circ_c_check[14, 15] = -0.267570088225568
    circ_c_check[15, 15] = 0.231097918395994
    
    check_matrix(circ_c_matrix, circ_c_check, 10 * eps)
    print("Test C passed!")


def test_frqi_circuit_float():
    """Test with float32 precision"""
    # You can add specific float32 tests here if needed
    print("Float32 test - use numpy.float32 for specific precision")


def test_frqi_circuit_double():
    """Test with float64 precision (default)"""
    test_compressed_frqi_circuit_case_a()
    test_compressed_frqi_circuit_case_b()
    test_compressed_frqi_circuit_case_c()


if __name__ == "__main__":
    # Run with pytest: pytest test_frqi.py -v
    # Or run directly:
    print("Running FRQI Circuit Tests...\n")
    test_compressed_frqi_circuit_case_a()
    test_compressed_frqi_circuit_case_b()
    test_compressed_frqi_circuit_case_c()
    
    print("\n✅ All tests passed successfully!")