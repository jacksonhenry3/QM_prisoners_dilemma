import numpy as np
from scipy.linalg import expm

# Note:
# - For two qubits, the joint state basis we are using is np.array([|CC>, |CD>, |DC>, |DD>])
# - expm computes the matrix exponential of a given square matrix. (ask if you dont know what that means)
# - np.kron(A, B) is the Kronecker product, representing the tensor product of two operators or states.

# Please feel free to ask questions
# I'd also request that you keep approximate track of how much time this takes.

C = np.array([[1], [0]]) # |C>
D = np.array([[0], [1]]) # |D>

# U(θ, φ)
def U(theta: float, phi: float) -> np.ndarray:
    uni = np.array([[np.exp(1j * phi) * np.cos(theta/2), np.sin(theta/2)],
                    [-np.sin(theta/2), np.exp(-1j * phi) * np.cos(theta/2)]])
    return uni
#done in roughly 19 minutes; most of the time was just spent looking up syntax

#Define Operators/strategies
C_op = U(0, 0)
D_op = U(np.pi, 0)
Q_op = U(0, np.pi/2)

#gamma within the range 0 -> pi/2; entanglement increases as gamma increases
#Get unitary Operator J; 9 minutes
def J(gamma):
    return expm(np.kron(-1j*gamma*D_op, D_op/2))

# get final state vector; should be 4x1 array (complex)
def final_state(U_A: np.ndarray, U_B: np.ndarray, J: np.ndarray, initial_state: np.ndarray):
    """
    Compute the final state vector.
    This should return a 4x1 NumPy array (complex).
    """
    fs_vect = J.transpose() * np.kron(U_A, U_B) * J * inital_state
    return fs_vect
    #46 minutes; yet to output a 4x1 matrix, however does output a 4x4
print(final_state(C_op, D_op, J(0), np.kron(C, C))) #test that doesn't produce a 4x1

def payoff_A(final_state: np.ndarray) -> float:
    """
    Compute the payoff given the final_state as a 1D NumPy array.
    The result should be a floating-point number.
    """
    pass

# After implementing these, test with all classical strategies (C and D) to verify correctness.
