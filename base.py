
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import colors

# Note:
# - For two qubits, the joint state basis we are using is np.array([|CC>, |CD>, |DC>, |DD>])
# - expm computes the matrix exponential of a given square matrix. (ask if you dont know what that means)
# - np.kron(A, B) is the Kronecker product, representing the tensor product of two operators or states.

# Please feel free to ask questions
# I'd also request that you keep approximate track of how much time this takes.

C = np.array([1, 0]) # |C>
D = np.array([0, 1]) # |D>

def U(theta: float, phi: float):
    """
    Implement the unitary U(θ, φ) as a 2x2 NumPy array (complex).
    """
    uni = np.array([[np.exp(1j * phi) * np.cos(theta/2), np.sin(theta/2)],
                    [-np.sin(theta/2), np.exp(-1j * phi) * np.cos(theta/2)]])
    return uni

#Define Operators/strategieskron()
C_op = U(0, 0)
D_op = U(np.pi, 0)
Q_op = U(0, np.pi/2)

#gamma within the range 0 -> pi/2; entanglement increases as gamma increases
#Get unitary Operator J; (9 minutes)
def J(gamma):
    return expm(np.kron(-1j*gamma*D_op, D_op/2))

def final_state(U_A: np.ndarray, U_B: np.ndarray, J: np.ndarray, initial_state: np.ndarray):
    """
    Compute the final state vector.
    This should return a 4x1 NumPy array (complex).
    """
    fs_vect = (J.conj().transpose() @ np.kron(U_A, U_B) @ J) @ initial_state
    return fs_vect

#Alice's payoff; return a float
def payoff_A(final_state: np.ndarray):
    payoff_vect1 = np.array([3, 0, 5, 1])
    payoff_vect2 = np.abs(final_state)**2
    pay_A = np.dot(payoff_vect1, payoff_vect2)
    return pay_A

def payoff_Moves(Bob_move, Alice_move):
    init_state = np.kron(C, C)
    J1 = J(np.pi/2)
    fin_state = final_state(Alice_move, Bob_move, J1, init_state)
    return payoff_A(fin_state)

print(payoff_Moves(C_op, D_op))

# After implementing these, test with all classical strategies (C and D) to verify correctness.

#print(payoff_A(final_state(C_op, D_op, J(0), np.kron(C,C))))
#print(payoff_A(final_state(D_op, D_op, J(0), np.kron(C,C))))
#print(payoff_A(final_state(C_op, C_op, J(0), np.kron(C,C))))
#print(payoff_A(final_state(D_op, C_op, J(0), np.kron(C,C))))
#print(np.kron(C, C))

"""
Compute the negative of Alice's payoff for given strategy parameters and an opponent's move.

Parameters:
    params (list or tuple): A pair [theta, phi] defining Alice's unitary strategy U(theta, phi).
    opponent_move (np.ndarray): Bob's strategy represented as a 2x2 unitary matrix.

Returns:
    float: The negative of Alice's expected payoff computed from the final state of the game.
    """

def negative_alice_payoff(params, opponent_move):
    # Extract strategy parameters: theta and phi for Alice's move
    theta, phi = params[0], params[1]

    # Compute Alice's unitary strategy using the given parameters
    alice_move = U(theta, phi)

    # Maximum entanglement operator (gamma = π/2)
    entanglement_operator = J(np.pi / 2)

    # Initial state: tensor product of |C> for Alice and |C> for Bob
    initial_state = np.kron(C, C)

    # Compute the final state after applying entanglement, strategies, and disentanglement
    final = final_state(alice_move, opponent_move,
                        entanglement_operator, initial_state)

    # Calculate Alice's expected payoff from the final state
    alice_payoff = payoff_A(final)

    # Return the negative payoff (useful for minimization routines)
    return -alice_payoff

    """
    Optimize Alice's strategy given an opponent's move by minimizing the negative payoff.

    Parameters:
        initial_guess (list): Initial guess [theta, phi] for the optimization.
        opponent_move (np.ndarray): Opponent's move represented as a 2x2 unitary matrix.

    Returns:
        tuple: Optimized strategy parameters (theta, phi) from the minimization.

    Process:
        - Uses 'minimize' to find the strategy that maximizes Alice's payoff
          (by minimizing the negative payoff function negative_alice_payoff).
        - The search is bounded: theta ∈ [0, π] and phi ∈ [0, π/2].
    """


def optimize_strategy(initial_guess, opponent_move):
    # Boundaries for theta and phi
    bounds = [(0, np.pi), (0, np.pi / 2)]

    # Perform optimization to minimize negative payoff function negative_alice_payoff
    result = minimize(negative_alice_payoff, initial_guess,
                      args=(opponent_move,), bounds=bounds)

    # Extract the optimized strategy parameters
    theta_opt, phi_opt = result.x
    return theta_opt, phi_opt

# tuple[0] = bob's move; Tuple[1] = possible response; Tuple[2] = J
def best_response(tuple):
    init_state = np.kron(C,C)
    fin_state = final_state(tuple[1], tuple[0], tuple[2], init_state)
    return ((payoff_A(fin_state)), tuple[0])

pi = np.pi
cos = np.cos
sin = np.sin
def dihedral_moves(n):
    #r_lst = []
    #s_lst = []
    lst = []
    for k in range(0, n-1):
        r = np.array([[cos((2*pi*k)/n), -sin((2*pi*k)/n)],
                     [sin((2*pi*k)/n), cos((2*pi*k)/n)]])
        s = np.array([[cos((2 * pi * k) / n), sin((2 * pi * k) / n)],
                      [sin((2 * pi * k) / n), -cos((2 * pi * k) / n)]])
        lst.append(r)
        lst.append(s)
    return lst

if __name__ == '__main__':

    bob_moves = dihedral_moves(4)
    #alice_moves = dihedral_moves(4)
    #initarray = np.zeros((8,8))
    #for (i, bob_moves) in enumerate(bob_moves):
    #    for (j, alice_moves) in enumerate(alice_moves):
    #        p = payoff_Moves(bob_moves, alice_moves)
    #        initarray[i, j] = p

    print(payoff_Moves(bob_moves[1], bob_moves[0]))

    with Pool(1) as p:
        bob_m = C_op
        bob_m2 = D_op
        G = J(0)
        lst = (bob_m, C_op, G), (bob_m, D_op, G), (bob_m2, C_op, G), (bob_m2, D_op, G)
        lst2 = (p.map(best_response, lst))
        #print(lst2)
        #print(lst2[2][0])

        #colors_list = ['#F0FF33', '#FFAC33', '#FF5733', '#A5FF33']
        #cmap = colors.ListedColormap(colors_list)
        mat = initarray
        plt.imshow(mat, cmap = 'autumn')
        #plt.xticks(range(len(mat.columns)))
        #plt.yticks(range(len(mat.columns)))
        plt.colorbar()
        plt.show()

