import numpy as np
from scipy.linalg import expm

# Note:
# - For two qubits, the joint state basis we are using is np.array([|CC>, |CD>, |DC>, |DD>])
# - expm computes the matrix exponential of a given square matrix. (ask if you dont know what that means)
# - np.kron(A, B) is the Kronecker product, representing the tensor product of two operators or states.

# Please feel free to ask questions
# I'd also request that you keep approximate track of how much time this takes.

C = np.array([1, 0]) # |C>
D = np.array([0, 1]) # |D>

# U(θ, φ); return a 2x2 nparray
def U(theta: float, phi: float):
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

#get final state vector; should return 4x1 array (complex)
#48 minutes; @ not * and forgot conj()
def final_state(U_A: np.ndarray, U_B: np.ndarray, J: np.ndarray, initial_state: np.ndarray):
    fs_vect = J.conj().transpose() @ np.kron(U_A, U_B) @ J @ initial_state
    return fs_vect

#Alice's payoff; return a float
#24 minutes;
def payoff_A(final_state: np.ndarray):
    r = 3 #(“reward”)
    p = 1 #(“punishment”)
    t = 5 #(“temptation”)
    s = 0 #(“sucker’s pay-off ”).
    PCC = pow(np.kron(C, C).transpose() @ final_state, 2)
    PDD = pow(np.kron(D, D).transpose() @ final_state, 2)
    PDC = pow(np.kron(D, C).transpose() @ final_state, 2)
    PCD = pow(np.kron(C, D).transpose() @ final_state, 2)
    pay_A = r*PCC + p*PDD + t*PDC + s*PCD
    return pay_A
    

# After implementing these, test with all classical strategies (C and D) to verify correctness.
# Tests; 5 minutes
print(payoff_A(final_state(C_op, D_op, J(0), np.kron(C,C)))) #Alice Cooperate, Bob Defect -> (1.1248198369963932e-32+0j); essentially 0
print(payoff_A(final_state(D_op, D_op, J(0), np.kron(C,C)))) #Alice Defect, Bob Defect -> (1+0j);
print(payoff_A(final_state(C_op, C_op, J(0), np.kron(C,C)))) #Alice Cooperate, Bob Cooperate -> (3+0j);
print(payoff_A(final_state(D_op, C_op, J(0), np.kron(C,C)))) #Alice Defect, Bob Cooperate -> (5+0j);

