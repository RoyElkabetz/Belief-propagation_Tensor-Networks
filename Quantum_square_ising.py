import numpy as np
import copy as cp
import gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import ncon



h = np.linspace(0., 4., num=50)
time_to_converge = np.zeros((len(h)))
mz_matrix_TN = np.zeros((2, 2, len(h)))

E = []
mx = []
mz = []
mx_exact = []
mz_exact = []

d = 8
p = 2
D_max = d
J = 1


imat = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1]])

smat = np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                 [1, 2, 0, 0, 3, 4, 0, 0],
                 [0, 0, 1, 2, 0, 0, 3, 4],
                 [0, 0, 0, 0, 1, 2, 3, 4]])



T0 = np.random.rand(p, d, d, d, d)
T1 = np.random.rand(p, d, d, d, d)
T2 = np.random.rand(p, d, d, d, d)
T3 = np.random.rand(p, d, d, d, d)

TT = [T0, T1, T2, T3]

LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1, 0.01, 0.001, 0.0001]
iterations = 100



for ss in range(h.shape[0]):
    T0 = np.random.rand(p, d, d, d, d)
    T1 = np.random.rand(p, d, d, d, d)
    T2 = np.random.rand(p, d, d, d, d)
    T3 = np.random.rand(p, d, d, d, d)

    TT = [T0, T1, T2, T3]

    LL = []
    for i in range(imat.shape[1]):
        LL.append(np.ones(d, dtype=float) / d)

    hij = -J * np.kron(pauli_z, pauli_z) - 0.25 * h[ss] * (np.kron(np.eye(p), pauli_x) + np.kron(pauli_x, np.eye(p)))
    hij_energy_operator = np.reshape(cp.deepcopy(hij), (p, p, p, p))
    hij = np.reshape(hij, [p ** 2, p ** 2])
    unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]

    counter = 0
    for i in range(len(t_list)):
        flag = 0
        for j in range(iterations):
            counter += 2
            print('h, i, j = ', h[ss], ss, i, j)
            TT1, LL1 = su.simple_update(cp.deepcopy(TT), cp.deepcopy(LL), unitary[i], imat, smat, D_max)
            TT2, LL2 = su.simple_update(cp.deepcopy(TT1), cp.deepcopy(LL1), unitary[i], imat, smat, D_max)

            energy1 = su.energy_per_site(cp.deepcopy(TT1), cp.deepcopy(LL1), imat, smat, hij_energy_operator)
            energy2 = su.energy_per_site(cp.deepcopy(TT2), cp.deepcopy(LL2), imat, smat, hij_energy_operator)

            if np.abs(energy1 - energy2) < 1e-8:
                flag = 1
                break
            else:
                TT = cp.deepcopy(TT2)
                LL = cp.deepcopy(LL2)
        if flag:
            flag = 0
            break
    time_to_converge[ss] = counter
    mx.append(su.magnetization(cp.deepcopy(TT), cp.deepcopy(LL), imat, smat, pauli_x))
    mz.append(su.magnetization(cp.deepcopy(TT), cp.deepcopy(LL), imat, smat, pauli_z))
    E.append(su.energy_per_site(cp.deepcopy(TT), cp.deepcopy(LL), imat, smat, hij_energy_operator))
    mmz = 0
    mmx = 0
    for jj in range(smat.shape[0]):
        zT_list, zidx_list = nlg.ncon_list_generator(TT, LL, smat, pauli_z, jj)
        zT_list_n, zidx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), jj)
        zz = ncon.ncon(zT_list, zidx_list) / ncon.ncon(zT_list_n, zidx_list_n)

        xT_list, xidx_list = nlg.ncon_list_generator(TT, LL, smat, pauli_x, jj)
        xT_list_n, xidx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), jj)
        xx = ncon.ncon(xT_list, xidx_list) / ncon.ncon(xT_list_n, xidx_list_n)
        mmz += zz
        mmx += xx
    mz_exact.append(mmz / smat.shape[0])
    mx_exact.append(mmx / smat.shape[0])
    print('Mx_exact, Mz_exact', mx_exact[ss], mz_exact[ss])
    print('E, Mx, Mz: ', E[ss], mx[ss], mz[ss])

    # Magnetization matrix vs h
    #kk = 0
    #for ii in range(2):
    #    for jj in range(2):
    #        mz_matrix_TN[ii, jj, ss] = su.single_tensor_expectation(np.int(2 * ii + jj), TT, LL, imat, smat, pauli_z)
    #        kk += 1



plt.figure()
plt.title('2D Quantum Ising Model in a transverse field at Dmax = ' + str(D_max))
plt.subplot()
color = 'tab:red'
plt.xlabel('h')
plt.ylabel('Energy per site', color=color)
plt.plot(h, E, color=color)
plt.grid()
plt.tick_params(axis='y', labelcolor=color)
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('# of iterations for energy convergence', color=color)  # we already handled the x-label with ax1
plt.plot(h, time_to_converge, color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()
'''
plt.figure()
plt.plot(h, E, 'o')
plt.xlabel('h')
plt.ylabel('Energy')
plt.grid()
plt.show()
'''
plt.figure()
plt.plot(h, mx, 'o')
plt.plot(h, np.abs(np.array(mz)), 'o')
plt.plot(h, mx_exact, 'o')
plt.plot(h, np.abs(np.array(mz_exact)), 'o')
plt.title('magnetization vs h at Dmax = ' + str(D_max))
plt.xlabel('h')
plt.ylabel('Magnetization')
plt.legend(['mx', '|mz|', 'mx exact', '|mz| exact'])
plt.grid()
plt.show()

'''
for i in range(len(h)):
    plt.figure()
    plt.matshow(mz_matrix_TN[:, :, i])
    plt.title('mz magnetizations at each site for h = ' + str(h[i]))
    plt.show()
'''