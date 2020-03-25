import numpy as np
import matplotlib.pyplot as plt
import DEnFG as denfg
import time
import Tensor_Network_contraction as tnc
import copy as cp

'THIS CODE IS SIMULATING A HONEYCOMB AKLT STATE'

# Initialize
t_max = 30
n = 16
l = 2 * np.sqrt(n)
aklt_spin_dim = 4
spin_half_dim = 2
c = 1. / np.sqrt(3)
singlet = np.array([[0., 1.], [-1., 0.]]) / np.sqrt(2)
'''
tensor = np.array([[[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
                   [[[0, c], [c, 0]], [[c, 0], [0, 0]]],
                   [[[0, 0], [0, c]], [[0, c], [c, 0]]],
                   [[[0, 0], [0, 0]], [[0, 0], [0, 1]]]])
'''
tensor = np.random.rand(4, 2, 2, 2) + np.random.rand(4, 2, 2, 2) * 1j

physical_spins = []
virtual_spins = []
for i in range(n):
    physical_spins.append('n' + str(i))
for i in range(n * 5 / 2):
    virtual_spins.append('n' + str(i + n))

# Constructing the DEnFG
graph = denfg.Graph()

# Adding nodes
for i in range(n):
    graph.add_node(aklt_spin_dim, physical_spins[i])
for i in range(n * 5 / 2):
    graph.add_node(2, virtual_spins[i])

# Adding factors
for i in range(2 * n):
    k = np.int(i / l)  # number of row
    j = np.mod(i, l)  # site # in row
    if np.mod(j, 2):  # singlet factors
        factor_neighbors = {virtual_spins[np.int(j - 1 + k * l)]: 0,
                            virtual_spins[np.int(np.mod(j, l) + k * l)]: 1}
        graph.add_factor(factor_neighbors, singlet)

    else:  # spin factors
        if np.mod(k, 2):  # uneven rows
            if np.mod(j / 2, 2):  # downward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + k * l / 4 + np.mod(j + 1, 3))]: 1,
                                    virtual_spins[i - 1]: 2,
                                    virtual_spins[i]: 3}
                graph.add_factor(factor_neighbors, tensor)
            else:  # upward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + (k - 1) * l / 4 + np.mod(j, 3))]: 1,
                                    virtual_spins[i]: 2,
                                    virtual_spins[np.int(l * k + np.mod(j + l - 1, l))]: 3}
                graph.add_factor(factor_neighbors, tensor)

        else:  # even rows
            if np.mod(j / 2, 2):  # upward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + (l / 2 - 1 - k) * l / 4 + np.mod(j + 1, 3))]: 1,
                                    virtual_spins[i]: 2,
                                    virtual_spins[i - 1]: 3}
                graph.add_factor(factor_neighbors, tensor)
            else:  # downward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + k * l / 4 + np.mod(j, 3))]: 1,
                                    virtual_spins[np.int(l * k + np.mod(j + l - 1, l))]: 2,
                                    virtual_spins[i]: 3}
                graph.add_factor(factor_neighbors, tensor)


# Belief Propagation implementation

z_operator = np.array([[3. / 2, 0, 0, 0],
                       [0, 1. / 2, 0, 0],
                       [0, 0, -1. / 2, 0],
                       [0, 0, 0, -3. / 2]])
x_operator = np.array([[0, np.sqrt(3), 0, 0],
                       [np.sqrt(3), 0, 2, 0],
                       [0, 2, 0, np.sqrt(3)],
                       [0, 0, np.sqrt(3), 0]])
i_operator = np.eye(4)
'''
z_operator = np.array([[1, 0], [0, -1]])
x_operator = np.array([[0, 1], [1, 0]])
'''
z_expectation = np.zeros((n, t_max), dtype=complex)
x_expectation = np.zeros((n, t_max), dtype=complex)
i_expectation = np.zeros((n, t_max), dtype=complex)

for t in range(t_max):
    s = time.time()
    graph.sum_product(t, 1e-6)
    graph.calc_node_belief()
    for m in range(n):
        z_expectation[m, t] = np.trace(np.matmul(graph.node_belief[physical_spins[m]], z_operator))
        x_expectation[m, t] = np.trace(np.matmul(graph.node_belief[physical_spins[m]], x_operator))
        i_expectation[m, t] = np.trace(np.matmul(graph.node_belief[physical_spins[m]], i_operator))
    e = time.time()
    print(t, 'time = ', e - s)

# plotting data
i = 0
plt.figure()
plt.plot(range(t_max), np.array(z_expectation[i, :]), 'o')
plt.plot(range(t_max), np.array(x_expectation[i, :]), 'o')
plt.plot(range(t_max), np.array(i_expectation[i, :]), 'o')

plt.xticks(range(0, t_max, 2))
plt.ylabel('$m$')
#plt.ylim([-1, 1])
plt.xlabel('# of BP iterations')
plt.legend(['$<\sigma_z(' + str(i) + ')>_{DE-NFG}$', '$<\sigma_x(' + str(i) + ')>_{DE-NFG}$', '$<I(' + str(i) + ')>_{DE-NFG}$'])
plt.grid()
plt.show()


'''
leg = []
plt.figure()
for i in range(n):
    leg.append('$<\sigma_z(' + str(i) + ')>_{DE-NFG}$')
    plt.plot(range(t_max), np.array(expectation[i, :]), 'o')

plt.xticks(range(0, t_max, 2))
plt.ylabel('$m_z$')
plt.xlabel('# of BP iterations')
#plt.legend(leg)
plt.grid()
plt.show()
'''

# playing with singlet orientation
'''
        if np.mod(k, 2):
            if np.mod(j + 1, 2):
                factor_neighbors = {virtual_spins[np.int(j - 1 + k * l)]: 1,
                                    virtual_spins[np.int(np.mod(j, l) + k * l)]: 0}
                graph.add_factor(factor_neighbors, singlet)
            else:
                factor_neighbors = {virtual_spins[np.int(j - 1 + k * l)]: 0,
                                    virtual_spins[np.int(np.mod(j, l) + k * l)]: 1}
                graph.add_factor(factor_neighbors, singlet)
        else:
            if np.mod(j + 1, 2):
                factor_neighbors = {virtual_spins[np.int(j - 1 + k * l)]: 0,
                                    virtual_spins[np.int(np.mod(j, l) + k * l)]: 1}
                graph.add_factor(factor_neighbors, singlet)
            else:
                factor_neighbors = {virtual_spins[np.int(j - 1 + k * l)]: 1,
                                    virtual_spins[np.int(np.mod(j, l) + k * l)]: 0}
                graph.add_factor(factor_neighbors, singlet)
        '''

# Generating the TN structure and index matrices
num_of_tensors = len(graph.factors.keys())
structure_mat = np.zeros((num_of_tensors, num_of_tensors), dtype=int)
index_mat = np.zeros((num_of_tensors, num_of_tensors), dtype=int)
tensors_list = []
tensors_name = []
tensors_list_idx = []
indices_list = []
second_indices_list = []

for item1 in graph.factors.keys():
    tensors_list.append(graph.factors[item1][1])
    tensors_list_idx.append(graph.factors[item1][2])
    tensors_name.append(item1)
    neighbors1 = graph.factors[item1][0]
    for item2 in graph.factors.keys():
        if item1 == item2:
            continue
        neighbors2 = graph.factors[item2][0]
        for object1 in neighbors1.keys():
            for object2 in neighbors2.keys():
                if object1 == object2:
                    structure_mat[graph.factors[item1][2], graph.factors[item2][2]] = 1
                    index_mat[graph.factors[item1][2], graph.factors[item2][2]] = graph.nodes[object1][2]

# Plotting the structure and index matrices
plt.figure()
plt.imshow(structure_mat)
plt.show()

plt.figure()
plt.imshow(index_mat)
plt.show()

# Constructing lists for scon (TN contraction function)
idx = np.max(index_mat) + 1
idx_new = np.copy(idx)
for i in range(structure_mat.shape[0]):
    indices_list.append(index_mat[i, np.nonzero(index_mat[:, i])])
    second_indices_list.append(index_mat[i, np.nonzero(index_mat[:, i])] + idx_new)
    if not np.mod(i, 2):
        indices_list[i] = np.insert(indices_list[i], 0, idx).tolist()
        second_indices_list[i] = np.insert(second_indices_list[i], 0, idx).tolist()
        idx += 1
    else:
        indices_list[i] = indices_list[i][0].tolist()
        second_indices_list[i] = second_indices_list[i][0].tolist()

ordered_tensors_list = [0] * len(tensors_list)
ordered_tensors_name = [0] * len(tensors_list)
second_ordered_tensors_list = [0] * len(tensors_list)

for i in range(len(ordered_tensors_list)):
    ordered_tensors_list[tensors_list_idx[i]] = cp.copy(tensors_list[i])
    ordered_tensors_name[tensors_list_idx[i]] = cp.copy(tensors_name[i])
    second_ordered_tensors_list[tensors_list_idx[i]] = cp.copy(np.conj(tensors_list[i]))
'''
for i in range(len(tensors_list)):
    print(ordered_tensors_name[i], ordered_tensors_list[i].shape, ' - ', indices_list[i])
    print(ordered_tensors_name[i], second_ordered_tensors_list[i].shape, ' - ', second_indices_list[i])
'''

final_indices_list = []
final_tensors_list = []
final_tensors_name = []

for i in range(len(indices_list)):
    final_indices_list.append(indices_list[i])
    final_indices_list.append(second_indices_list[i])
    final_tensors_list.append(ordered_tensors_list[i])
    final_tensors_list.append(second_ordered_tensors_list[i])
    final_tensors_name.append(ordered_tensors_name[i])
    final_tensors_name.append(ordered_tensors_name[i] + '*')


for i in range(len(final_tensors_list)):
    print(final_tensors_name[i], final_tensors_list[i].shape, ' - ', final_indices_list[i])
'''
contraction_order = [56, 16, 72, 17, 73, 57, 18, 74, 19, 75, 58, 20, 76, 21, 77, 59, 22, 78, 23, 79,
                     48, 104, 60, 49, 105, 62, 24, 80, 25, 81, 61, 26, 82, 27, 83, 28, 84, 29, 85, 63, 30, 86, 31, 87,
                     50, 106, 65, 51, 107, 67, 34, 90, 35, 91, 66, 36, 92, 37, 93, 38, 94, 39, 95, 64, 32, 88, 33, 89,
                     52, 108, 68, 53, 109, 70, 40, 96, 41, 97, 69, 42, 98, 43, 99, 44, 100, 45, 101, 71, 46, 102, 47, 103, 54, 110, 55, 111]
'''

contraction_order = [56, 16, 72, 17, 73, 57, 18, 74, 19, 75, 58, 20, 76, 21, 77, 59, 22, 78, 23, 79,
                     48, 104, 60, 49, 105, 62, 24, 80, 25, 81, 61, 26, 82, 27, 83, 28, 84, 29, 85, 63, 30, 86, 31, 87,
                     50, 106, 65, 51, 107, 67, 34, 90, 35, 91, 66, 36, 92, 37, 93, 38, 94, 39, 95, 64, 32, 88, 33, 89,
                     52, 108, 68, 53, 109, 70, 40, 96, 41, 97, 69, 42, 98, 43, 99, 44, 100, 45, 101, 71, 46, 102, 47, 103, 54, 110, 55, 111]
# Contracting the Tensor Network for normalization
tn_normalization = tnc.scon(final_tensors_list, tuple(final_indices_list), contraction_order)

# Multiplying the first tensor with an operator
final_tensors_list[0] = np.einsum(final_tensors_list[0], [0, 1, 2, 3], z_operator, [0, 4], [4, 1, 2, 3])


# Contracting the Tensor Network with an operator
psi = tnc.scon(final_tensors_list, tuple(final_indices_list), contraction_order)

operator_expectetion = psi / tn_normalization
print('\n')
print('tn_normalization = ', tn_normalization)
print('psi = ', psi)
print('operator_expectetion = ', operator_expectetion)
print('\n')


# old data
'''
z_operator:
    aklt tensor
    ('tn_normalization = ', array(0.00484694))
    ('psi = ', array(-1.62630326e-19))
    ('operator_expectetion = ', '-3.3553215520575584e-17')
    bp_expectation = 0

    
    random tensor
    ('tn_normalization = ', array(0.94518942))
    ('psi = ', array(-0.27503748))
    ('operator_expectetion = ', '-0.290986622678143')
    bp_expectation = -0.31419954558956675
    
i_operator:
    aklt tensor
    ('tn_normalization = ', array(0.00484694))
    ('psi = ', array(0.00484694))
    ('operator_expectetion = ', '1.0')
    bp_expectation = 1
    
    random tensor
    ('tn_normalization = ', array(1.09398021))
    ('psi = ', array(1.09398021))
    ('operator_expectetion = ', '1.0')
    bp_expectation = 1

'''





