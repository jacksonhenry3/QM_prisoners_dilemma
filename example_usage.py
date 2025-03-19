import numpy as np
import Quantum_Prisoners_Dillema as QPD


# Some common strategy spaces
sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
SU2_bassis = [sigma1, sigma2, sigma3]
SU2 = StrategySpace(SU2_bassis, is_discrete=False)

d4 = dihedral_group(4)
D = np.array([[0,-1],[1,0]])
J = QPD.J(D,D)

symmetric_d4_qpd = QPD.QuantumPrisonersDilema(J,d4)

symmetric_d4_qpd.plot()
