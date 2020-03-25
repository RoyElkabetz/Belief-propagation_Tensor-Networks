import numpy as np
import copy as cp
import gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt

#d_vec = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
d_vec = [3]

E = []
d_and_t = np.zeros((2, len(d_vec)))

d = 2
p = 2
J = -1


imat = np.array([[1, 1],
                 [1, 1]])

smat = np.array([[2, 1],
                 [2, 1]])

T0 = np.random.rand(p, d, d)
T1 = np.random.rand(p, d, d)
TT = [T0, T1]

LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1]
iterations = 20
heisenberg = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(heisenberg, (p, p, p, p))
hij_energy_term = cp.deepcopy(hij)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]

for ss in range(len(d_vec)):


    D_max = d_vec[ss]

    counter = 0
    for i in range(len(t_list)):
        flag = 0
        for j in range(iterations):
            counter += 1
            print('D_max, ss, i, j = ',D_max, ss, i, j)
            TT1, LL1 = su.simple_update(cp.deepcopy(TT), cp.deepcopy(LL), unitary[i], imat, smat, D_max)
            TT2, LL2 = su.simple_update(cp.deepcopy(TT1), cp.deepcopy(LL1), unitary[i], imat, smat, D_max)

            energy1 = su.energy_per_site(TT1, LL1, imat, smat, hij_energy_term)
            energy2 = su.energy_per_site(TT2, LL2, imat, smat, hij_energy_term)

            if np.abs(energy1 - energy2) < 1e-10:
                flag = 1
                break
            else:
                TT = cp.deepcopy(TT2)
                LL = cp.deepcopy(LL2)
        if flag:
            flag = 0
            break
    d_and_t[:, ss] = np.array([D_max, counter])
    E.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
    print(E[ss])

dd = np.zeros(len(d_vec))
for i in range(len(dd)):
    dd[i] = np.float(d_vec[i]) ** (-1)

plt.figure()
plt.title('AFH chain energy')
plt.plot(dd, E, 'o')
plt.xlabel('D_max')
plt.ylabel('normalized energy')
plt.grid()
plt.show()

plt.figure()
plt.title('AFH chain iterations')
plt.plot(d_and_t[0, :], d_and_t[1, :], 'o')
plt.xlabel('D_max')
plt.ylabel('# of ITE iters until converged')
plt.grid()
plt.show()