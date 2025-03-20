import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from numpy.linalg import matrix_power, det
import itertools
from functools import reduce


class StrategySpace:
    def __init__(self, generators: list[np.ndarray], is_discrete=True):
        if is_discrete:
            for generator in generators:
                # Enforce unitarity
                assert np.all(np.isclose(generator@generator.conjugate().T, np.eye(2))
                              ), "all generators must be unitary for discrete groups."

                # enforce that it has det 1 (special)
                assert np.isclose(np.abs(det(
                    generator)), 1), "All generators must have determinant 1 or -1 for discrete groups"
        else:
            for generator in generators:

                # Enforce hermiticity
                assert np.all(np.isclose(generator, generator.conjugate(
                ).T)), "All generators must be hermitian for continuous groups"

                # enforce traceless
                assert np.isclose(
                    generator.trace(), 0), "All generators must be traceless for continuous groups"
        self.generators = generators
        self.is_discrete = is_discrete

    def get_value(self, parameters: list[float]):
        if len(parameters) != len(self.generators):
            raise ValueError(
                f"Expected {len(self.generators)} parameters, got {len(parameters)}.")
        if self.is_discrete:
            assert np.issubdtype(parameters.dtype, np.integer)
            matrices = [matrix_power(gen,pow) for (gen, pow)
                        in zip(self.generators, parameters)]
            return reduce(np.dot, matrices)
        else:
            return expm(1j*np.sum([gen*scale for (gen, scale) in zip(self.generators, parameters)], axis=0))

    def all_elements(self):
        if not self.is_discrete:
            raise ValueError(
                "can't realize all elements of a continous group")

        # Convert a numpy array to a hashable canonical tuple representation.
        def canonical(mat):
            # THe rounding here is so that elements are identified even with float errors.
            return tuple(map(tuple, np.round(mat, 5)))

            # Assume all generators are square matrices; start with the identity.
        identity = np.eye(
            self.generators[0].shape[0], dtype=self.generators[0].dtype)

        group = {canonical(identity)}
        for g in self.generators:
            group.add(canonical(g))

            changed = True
            while changed:
                changed = False
                new_elems = set()
                # Use itertools.product to generate pairwise products.
                for a, b in itertools.product(group, repeat=2):
                    prod = np.dot(np.array(a), np.array(b))
                    prod_can = canonical(prod)
                    if prod_can not in group:
                        new_elems.add(prod_can)
                    if new_elems:
                        group.update(new_elems)
                        changed = True
        return [np.array(g) for g in group]

def dihedral_group(n: int) -> StrategySpace:
    """
    because of the float operations, sometimes it may make sense to round the result to get integer values.
    """
    r_lst = []
    s_lst = []
    for k in range(0, n-1):
        r = np.array([[cos((2*pi*k)/n), -sin((2*pi*k)/n)],
                     [sin((2*pi*k)/n), cos((2*pi*k)/n)]])
        s = np.array([[cos((2 * pi * k) / n), sin((2 * pi * k) / n)],
                      [sin((2 * pi * k) / n), -cos((2 * pi * k) / n)]])
        r_lst.append(r)
        s_lst.append(s)

    return StrategySpace([r_lst, s_lst])



class QuantumPrisonersDilema:
    def __init__(self, EntanglementOperator: np.ndarray, strategy_space):
        self.J = EntanglementOperator
        self.strategy_space = strategy_space
        self.alice_payoff_list = np.array([3, 0, 5, 1])
        self.bob_payoff_list = np.array([3, 5, 0, 1])
        self.inital_state = np.kron(C, C)

    def play(self, alice_move: np.ndarray, bob_move: np.ndarray) -> np.ndarray:
        """
        Return the final state after alice and bob play their moves
        """
        fs_vect = (EntanglementOperator.conj().transpose() @ np.kron(alice_move, bob_move) @ EntanglementOperator) @ self.initial_state
        return fs_vect

    def _calculate_payoff(self, final_state: np.ndarray) -> tuple[float, float]:
        """
        Return a tuple of (alice payoff, bob payoff)
        """
        payoff_vect = np.abs(final_state)**2
        pay_A = np.dot(self.alice_payoff_list, payoff_vect)
        pay_B = np.dot(self.bob_payoff_list, payoff_vect)
        return [pay_A, pay_B]

    def payoff(self, alice_move: np.ndarray, bob_move: np.ndarray) -> tuple[float, float]:
        return self._calculate_payoff(self.play(alice_move, bob_move))

    def best_response(self, alice_move: np.ndarray) -> np.ndarray:
        """
        given alice_move return the best move for bob.

        Question: is best the largest difference in bobs favor? or the largest score for bob?
        """
        pass

    def find_pareto_optimum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        find a pair of moves that is pareto optimal

        A Pareto optimum is a state of resource allocation where no individual's situation can be improved without making at least one other individual worse off.
        """
        pass

    def find_nash_equilibrium(self) -> tuple[np.ndarray, np.ndarray]:
        """
        find a pair of moves the is a nash equilibrium

        A Nash equilibrium is a set of strategies where no player can improve their payoff by unilaterally changing their own strategy, assuming all other players' strategies remain constant. 
        """
        pass

    def plot(self):
        """
        If the self.strategy_space is discrete then plot a payoff matrix for all moves.

        If not, maybe something related to a cayleigh graph?
        """
        if self.strategy_space.is_discrete:
            strats = self.strategy_space.all_elements()
            payoff_matrix = np.array([[self.payoff(alice_strat, bob_strat)[
                                     0] for alice_strat in strats] for bob_strat in strats])
            plt.imshow(payoff_matrix)
            plt.show()

    def J(A:np.ndarray,B:np.ndarray) -> np.ndarray:
        gamma = np.pi/2
        """
        generate the entanglement operator from two input arrays
    
        This function should enforce the conditions on A and B
        """
        return expm(np.kron(-1j*gamma*A, B/2))
        pass
