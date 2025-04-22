import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from multiprocessing import Pool
from numpy.linalg import matrix_power, det
import itertools
from functools import reduce

#C = np.array([1, 0]) # |C>
#D = np.array([0, 1]) # |D>

def base_qpd():
    c_op = np.eye(2)
    d_op = np.array([[0, 1], [-1, 0]])
    return [c_op, d_op]

def cyclic_group(n: int):
    r = np.array([[np.cos(2*np.pi/n), -np.sin(2*np.pi/n)], [np.sin(2*np.pi/n), np.cos(2*np.pi/n)]])
    return [np.linalg.matrix_power(r,i) for i in range(0, n)]

def dihedral_group(n: int):
    """
    because of the float operations, sometimes it may make sense to round the result to get integer values.
    """
    c = cyclic_group(n)
    s = np.array([[0, 1],[1, 0]])
    c_prime = [i @ s for i in c]
    return c+c_prime

class QuantumPrisonersDilema:
    def __init__(self, EntanglementOperator: np.ndarray, strategy_space):
        self.J = EntanglementOperator
        self.strategy_space = strategy_space
        self.alice_payoff_list = np.array([3, 0, 5, 1])
        self.bob_payoff_list = np.array([3, 5, 0, 1])
        self.initial_state = np.array([1, 0, 0, 0])

    def play(self, alice_move: np.ndarray, bob_move: np.ndarray) -> np.ndarray:
        """
        Return the final state after alice and bob play their moves
        """
        fs_vect = (self.J.conj().transpose() @ np.kron(alice_move, bob_move) @ self.J) @ self.initial_state
        return fs_vect

    def _calculate_payoff(self, final_state: np.ndarray) -> tuple[float, float]:
        """
        Return a tuple of (alice payoff, bob payoff)
        """
        payoff_vect = np.abs(final_state)**2
        pay_A = np.dot(self.alice_payoff_list, payoff_vect)
        pay_B = np.dot(self.bob_payoff_list, payoff_vect)
        return (pay_A, pay_B)



    def payoff(self, alice_move: np.ndarray, bob_move: np.ndarray) -> tuple[float, float]:
        return self._calculate_payoff(self.play(alice_move, bob_move))

    def best_response(self, alice_move: np.ndarray) -> np.ndarray:
        """
        given alice_move return the best move for bob.

        Question: is best the largest difference in bobs favor? or the largest score for bob?
        """
        pass

    def find_pareto_optimums(self):
        """np.ndarray, np.ndarray)
        find a pair of moves that is pareto optimal

        A Pareto optimum is a state of resource allocation where no individual's situation can be improved without making at least one other individual worse off.
        """
        optimums = []
        Strats = self.strategy_space
        for i, s1 in enumerate(Strats):
            for j, s2 in enumerate(Strats):
                if self.is_pareto_optimal((s1,s2),(i, j)):
                    optimums.append((s1,s2))
        return optimums

    def is_pareto_optimal(self, currentstrats, indexes) -> bool:
        Strats = self.strategy_space
        initscore = self.payoff(alice_move=currentstrats[0], bob_move=currentstrats[1])
        for i, s1 in enumerate(Strats):
            for j, s2 in enumerate(Strats):
                compare = self.payoff(s1, s2)
                greaterA = compare[0] > initscore[0]
                greaterB = compare[1] > initscore[1]
                eqA =  compare[0] == initscore[0]
                eqB =  compare[1] == initscore[1]
                if (eqA and greaterB) or (eqB and greaterA):
                    return False
                notsameck = (i != indexes[0] or j != indexes[1])
                if greaterA and greaterB and notsameck:
                    return False
        return True

    def find_nash_equilibrium(self):
        """
        find a pair of moves the is a nash equilibrium

        A Nash equilibrium is a set of strategies where no player can improve their payoff by unilaterally changing their own strategy, assuming all other players' strategies remain constant.
        """
        equilibriums = []
        Strats = self.strategy_space
        for s1 in Strats:
            for s2 in Strats:
                if self.is_nash_equilibrium((s1, s2)):
                    equilibriums.append((s1, s2))
        return equilibriums

    def is_nash_equilibrium(self, currentstrats):
        Strats = self.strategy_space
        initscore = self.payoff(alice_move=currentstrats[0], bob_move=currentstrats[1])
        for s1 in Strats:
            compare = self.payoff(s1, currentstrats[1])
            greatereqA = compare[0] > initscore[0]
            if greatereqA:
                return False

        for s2 in Strats:
            compare = self.payoff(currentstrats[0], s2)
            greatereqB = compare[1] > initscore[1]
            if greatereqB:
                return False
        return True

    def plot(self):
        """
        If the self.strategy_space is discrete then plot a payoff matrix for all moves.

        If not, maybe something related to a cayleigh graph?
        """
        strats = self.strategy_space
        payoff_matrix = np.array([[self.payoff(alice_strat, bob_strat)[
        0] for alice_strat in strats] for bob_strat in strats])
        plt.imshow(payoff_matrix)
        plt.colorbar()
        plt.show()

    #A_param[0] = a; A_param[1] = x
    #B_param[0] = b; B_param[1] = y
    #get D_op by *_params = (0, 1)
def J(A_param, B_param, gamma) -> np.ndarray:
    #if gamma < 0 or gamma > np.pi/2:
     #   raise ValueError("Expected a gamma value between 0 and pi/2")
    #else:
    A = np.array([[A_param*1j,1],[-1,A_param*1j]])
    B = np.array([[B_param*1j,1],[-1,B_param*1j]])
    return expm(np.kron((-1j*gamma*A), B/2))

def find_best_ab(num_steps, start, stop, strats):
    best_a = 0.0
    best_b = 0.0
    best_gamma = 0.0
    best_scores = (0, 0)
    ab_range = np.linspace(start, stop, num_steps, True, True)
    gamma_range = np.linspace(0, np.pi/2, num_steps, True, True)
    for A in ab_range[0]: #A
        for B in ab_range[0]: #B

                    qpd = QuantumPrisonersDilema(J(A, B, g), strats)
                    nash_lst = qpd.find_nash_equilibrium()
                    if len(nash_lst) == 1:
                        scores = qpd.payoff(nash_lst[0][0], nash_lst[0][1])
                        if scores[0] > best_scores[0] or scores[1] > best_scores[1]:
                            best_scores = scores
                            best_a = A
                            best_b = B
                            best_gamma = g

    return best_a, best_b, best_gamma, best_scores
DD_QPD = QuantumPrisonersDilema(EntanglementOperator=J((9.84924623115578), (9.949748743718594), gamma=.4183527905534147), strategy_space= dihedral_group(4))
lst = DD_QPD.find_nash_equilibrium()
DD_QPD.plot()
for i in range(len(lst)):
    print(DD_QPD.payoff(lst[i][0], lst[i][1]))
print(lst)
print(dihedral_group(4))

#DD_QPD.find_nash_equilibrium()
#Weird_QPD = QuantumPrisonersDilema(EntanglementOperator=J((4/2, 2/2), (7/2, 8/2), gamma=(np.pi*4)/2), strategy_space= dihedral_group(2))
#Weird_QPD.plot()

rotation_QPD = QuantumPrisonersDilema(EntanglementOperator=J((0),(0), gamma= np.pi/2), strategy_space= cyclic_group(128))
#rotation_QPD.plot()
#[1] is C
#[5] is D
#C = DD_QPD.strategy_space.all_elements()[1]
#D = DD_QPD.strategy_space.all_elements()[5]
#E = DD_QPD.strategy_space.all_elements()
#print(DD_QPD.payoff(C, C))
#print(DD_QPD.payoff(C, D))
#print(DD_QPD.payoff(D, C))
#print(DD_QPD.payoff(D, D))
#print(E)
#print(np.round(J((4/2, 2/2), (7, 8), gamma=np.pi/2), 4))
#print((DD_QPD.find_pareto_optimums()))
#print(DD_QPD.find_nash_equilibrium())
#print(DD_QPD.payoff(DD_QPD.find_nash_equilibrium()[0][0],DD_QPD.find_nash_equilibrium()[0][1]))

#print(dihedral_group(4))
#print(cyclic_group(4))

print(find_best_ab(200, 0, 10, dihedral_group(4)))
