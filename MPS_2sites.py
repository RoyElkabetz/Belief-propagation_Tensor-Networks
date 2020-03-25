import numpy as np
import copy as cp
import gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt


d = 20
p = 2
D_max = d
J = -1

print('\n')
print('D_max = ', D_max)
print('\n')


'''
T0 = np.random.rand(p, d, d)
T1 = np.random.rand(p, d, d)
T2 = np.random.rand(p, d, d)
T3 = np.random.rand(p, d, d)
T4 = np.random.rand(p, d, d)
T5 = np.random.rand(p, d, d)
T6 = np.random.rand(p, d, d)
T7 = np.random.rand(p, d, d)
TT = [T0, T1, T2, T3, T4, T5, T6, T7]
imat = np.array([[1, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1]])

smat = np.array([[2, 0, 0, 0, 0, 0, 0, 1],
                 [1, 2, 0, 0, 0, 0, 0, 0],
                 [0, 1, 2, 0, 0, 0, 0, 0],
                 [0, 0, 1, 2, 0, 0, 0, 0],
                 [0, 0, 0, 1, 2, 0, 0, 0],
                 [0, 0, 0, 0, 1, 2, 0, 0],
                 [0, 0, 0, 0, 0, 1, 2, 0],
                 [0, 0, 0, 0, 0, 0, 1, 2]])
'''
'''
TT = [np.random.rand(p, d, d) for i in range(16)]
imat = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])

smat = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]])


'''
"""
T0 = np.random.rand(p, d, d)
T1 = np.random.rand(p, d, d)
T2 = np.random.rand(p, d, d)
TT = [T0, T1, T2]
imat = np.array([[1, 0, 1],
                 [1, 1, 0],
                 [0, 1, 1]])

smat = np.array([[2, 0, 1],
                 [1, 2, 0],
                 [0, 1, 2]])
"""

T0 = np.random.rand(p, d, d)
#T0 = np.arange(p * d * d).reshape(p, d, d)
T1 = np.random.rand(p, d, d)
#T1 = np.arange(p * d * d).reshape(p, d, d)

TT = [T0, T1]
imat = np.array([[1, 1],
                 [1, 1]])

smat = np.array([[2, 1],
                 [2, 1]])

LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)
    #LL.append(np.random.rand(d))

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x
'''
sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1.]])
sy = np.array([[0, -1j, 0.], [1j, 0, -1j], [0, 1j, 0.]]) / np.sqrt(2)
sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1., 0]]) / np.sqrt(2)

sz = np.array([[3. / 2, 0, 0, 0], [0, 1. / 2, 0, 0], [0, 0, -1. / 2, 0], [0, 0, 0, -3. / 2]])
sy = np.array([[0, np.sqrt(3), 0, 0], [-np.sqrt(3), 0, 2, 0], [0, -2, 0, np.sqrt(3)], [0, 0, -np.sqrt(3), 0]]) / 2j
sx = np.array([[0, np.sqrt(3), 0, 0], [np.sqrt(3), 0, 2, 0], [0, 2, 0, np.sqrt(3)], [0, 0, np.sqrt(3), 0]]) / 2
'''
t_list = np.exp(np.concatenate((np.linspace(-1, -3, 100), np.linspace(-3, -5, 100))))
heisenberg = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(cp.deepcopy(heisenberg), (p, p, p, p))
hij_perm = [0, 2, 1, 3]
hij_energy_term = cp.deepcopy(hij)
#hij = np.transpose(hij, hij_perm)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]
eye = np.reshape(np.eye(p * p), (p, p, p, p))

iterations = 1
energy = []
LL_in_time = np.zeros((len(LL), D_max, len(t_list) * iterations), dtype=float)
TT_in_time = np.zeros((len(TT), len(np.ravel(TT[0])), len(t_list) * iterations))
counter = 0
for i in range(len(t_list)):
    for j in range(iterations):
        print('t, iters = ', i, j)
        for k in range(len(LL)):
            LL_in_time[k, :, counter] = cp.deepcopy(LL[k])
        for l in range(len(TT)):
            TT_in_time[l, :, counter] = cp.deepcopy(np.ravel(TT[l]))
        TT_new, LL_new = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        energy.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
        counter += 1

        TT = cp.deepcopy(TT_new)
        LL = cp.deepcopy(LL_new)


for k in range(len(LL)):
    plt.figure()
    plt.title('lambda' + str(k) + ' values in time')
    plt.xlabel('t')
    for s in range(D_max):
        plt.plot(range(counter), LL_in_time[k, s, :], 'o')
    plt.grid()
    plt.ylim([0, 1])
    plt.show()

plt.figure()
plt.title('energy values')
plt.xlabel('t')
plt.plot(range(counter), energy, 'o')
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
