import numpy as np
import copy as cp
import simple_update_algorithm2 as su
from scipy import linalg
import matplotlib.pyplot as plt

d = 6
p = 2
D_max = d
J = 1.

T0 = np.random.rand(p, d, d, d)
T1 = cp.copy(T0)
T2 = cp.copy(T0)
T3 = cp.copy(T0)
T4 = cp.copy(T0)
T5 = cp.copy(T0)

TT = [T0, T1, T2, T3, T4, T5]


imat = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 1, 1, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 0],
                 [0, 0, 0, 0, 1, 0, 1, 0, 1],
                 [0, 0, 1, 0, 0, 0, 0, 1, 1]])

smat = np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 2, 3, 0, 0, 0, 0],
                 [0, 1, 0, 2, 0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 2, 3, 0],
                 [0, 0, 0, 0, 1, 0, 2, 0, 3],
                 [0, 0, 1, 0, 0, 0, 0, 2, 3]])

LL = []
for i in range(9):
    LL.append(np.ones(d, dtype=float) / d)

#sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1.]])
#sy = np.array([[0, -1j, 0.], [1j, 0, -1j], [0, 1j, 0.]]) / np.sqrt(2)
#sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1., 0]]) / np.sqrt(2)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = np.exp(np.array(np.linspace(-1, -10, 100)))
heisenberg = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(heisenberg, (p, p, p, p))
hij_perm = [0, 2, 1, 3]
hij_energy_term = cp.deepcopy(hij)
hij = np.transpose(hij, hij_perm)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]



iterations = 1
energy = []
LL_in_time = np.zeros((len(LL), D_max, len(t_list) * iterations), dtype=float)
TT_in_time = np.zeros((len(TT), len(np.ravel(TT[0])), len(t_list) * iterations))
counter = 0
for i in range(len(t_list)):
    for j in range(iterations):
        print('t, iters = ', i, j)
        for k in range(len(LL)):
            LL_in_time[k, :, counter] = LL[k]
        for l in range(len(TT)):
            TT_in_time[l, :, counter] = np.ravel(TT[l])
        TT, LL = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        #TT, LL = su.gauge_fix1(TT, LL, imat, smat)
        energy.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
        counter += 1
'''
for k in range(len(LL)):
    plt.figure()
    #plt.title('lambda' + str(k) + ' values in time')
    plt.xlabel('t')
    for s in range(D_max):
        plt.plot(range(len(t_list) * iterations), LL_in_time[k, s, :], 'o')
    plt.grid()
    plt.show()
'''
plt.figure()
plt.title('energy values')
plt.xlabel('t')
plt.plot(range(len(t_list) * iterations), energy, 'o')
plt.grid()
plt.show()

'''
for k in range(len(TT)):
    plt.figure()
    plt.title('T' + str(k) + ' entries values in time')
    plt.xlabel('t')
    for s in range(len(np.ravel(T0))):
        plt.plot(range(len(t_list) * iterations), TT_in_time[k, s, :], 'o')
    plt.grid()
    plt.show()
'''