def negative_alice_payoff(params, opponent_move):
    """
    Compute the negative of Alice's payoff for given strategy parameters and an opponent's move.

    Parameters:
        params (list or tuple): A pair [theta, phi] defining Alice's unitary strategy U(theta, phi).
        opponent_move (np.ndarray): Bob's strategy represented as a 2x2 unitary matrix.

    Returns:
        float: The negative of Alice's expected payoff computed from the final state of the game.

    Process:
        1. Construct Alice's strategy using U(theta, phi) with parameters from 'params'.
        2. Prepare the entanglement operator J at maximum entanglement (gamma = π/2).
        3. Define the initial state as a tensor product of two |C> states.
        4. Compute the final state vector by applying:
           a. The entanglement operator J to the initial state.
           b. The tensor product of Alice's and Bob's strategies.
           c. The inverse of the entanglement operator (via its conjugate transpose).
        5. Calculate Alice's payoff using the defined payoff function.
        6. Return the negative payoff (often used in optimization contexts where minimization is performed).
    """
    # Extract strategy parameters: theta and phi for Alice's move
    theta, phi = params[0], params[1]

    # Compute Alice's unitary strategy using the given parameters
    alice_move = U(theta, phi)

    # Maximum entanglement operator (gamma = π/2)
    entanglement_operator = J(np.pi/2)

    # Initial state: tensor product of |C> for Alice and |C> for Bob
    initial_state = np.kron(C, C)

    # Compute the final state after applying entanglement, strategies, and disentanglement
    final = final_state(alice_move, opponent_move,
                        entanglement_operator, initial_state)

    # Calculate Alice's expected payoff from the final state
    alice_payoff = payoff_A(final)

    # Return the negative payoff (useful for minimization routines)
    return -alice_payoff


def optimize_strategy(initial_guess, opponent_move):
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
    # Boundaries for theta and phi
    bounds = [(0, np.pi), (0, np.pi/2)]

    # Perform optimization to minimize negative payoff function negative_alice_payoff
    result = minimize(negative_alice_payoff, initial_guess,
                      args=(opponent_move,), bounds=bounds)

    # Extract the optimized strategy parameters
    theta_opt, phi_opt = result.x
    return theta_opt, phi_opt
