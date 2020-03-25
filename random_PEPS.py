import numpy as np
import matplotlib.pyplot as plt
import DEnFG as denfg
import time
import Tensor_Network_contraction as tnc
import copy as cp

'THIS CODE IS SIMULATING A HONEYCOMB RANDOM TENSORS STATE'

# Initialize
t_max = 30
n = 16
l = 2 * np.sqrt(n)
physical_dim = 2
virtual_dim = 2
c = 1. / np.sqrt(3)
bond_tensor = np.array([[0., 1.], [-1., 0.]]) / np.sqrt(2)
'''
tensor = np.array([[[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
                   [[[0, c], [c, 0]], [[c, 0], [0, 0]]],
                   [[[0, 0], [0, c]], [[0, c], [c, 0]]],
                   [[[0, 0], [0, 0]], [[0, 0], [0, 1]]]])
'''
physical_tensor = np.random.rand(physical_dim, virtual_dim, virtual_dim, virtual_dim) + np.random.rand(physical_dim, virtual_dim, virtual_dim, virtual_dim) * 1j

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
    graph.add_node(physical_dim, physical_spins[i])
for i in range(n * 5 / 2):
    graph.add_node(2, virtual_spins[i])

# Adding factors
for i in range(2 * n):
    k = np.int(i / l)  # number of row
    j = np.mod(i, l)  # site # in row
    if np.mod(j, 2):  # singlet factors
        factor_neighbors = {virtual_spins[np.int(j - 1 + k * l)]: 0,
                            virtual_spins[np.int(np.mod(j, l) + k * l)]: 1}
        graph.add_factor(factor_neighbors, bond_tensor)

    else:  # spin factors
        if np.mod(k, 2):  # uneven rows
            if np.mod(j / 2, 2):  # downward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + k * l / 4 + np.mod(j + 1, 3))]: 1,
                                    virtual_spins[i - 1]: 2,
                                    virtual_spins[i]: 3}
                graph.add_factor(factor_neighbors, physical_tensor)
            else:  # upward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + (k - 1) * l / 4 + np.mod(j, 3))]: 1,
                                    virtual_spins[i]: 2,
                                    virtual_spins[np.int(l * k + np.mod(j + l - 1, l))]: 3}
                graph.add_factor(factor_neighbors, physical_tensor)

        else:  # even rows
            if np.mod(j / 2, 2):  # upward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + (l / 2 - 1 - k) * l / 4 + np.mod(j + 1, 3))]: 1,
                                    virtual_spins[i]: 2,
                                    virtual_spins[i - 1]: 3}
                graph.add_factor(factor_neighbors, physical_tensor)
            else:  # downward leg factors
                factor_neighbors = {physical_spins[i / 2]: 0,
                                    virtual_spins[np.int(2 * n + k * l / 4 + np.mod(j, 3))]: 1,
                                    virtual_spins[np.int(l * k + np.mod(j + l - 1, l))]: 2,
                                    virtual_spins[i]: 3}
                graph.add_factor(factor_neighbors, physical_tensor)

# Generating the TN structure and index matrices
num_of_tensors = len(graph.factors.keys())
structure_mat = np.zeros((num_of_tensors, num_of_tensors), dtype=int)
index_mat = np.zeros((num_of_tensors, num_of_tensors), dtype=int)
tensors_list = []
tensors_name = []
tensors_list_idx = []
indices_list = []
#second_indices_list = []

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
    #second_indices_list.append(index_mat[i, np.nonzero(index_mat[:, i])] + idx_new)
    if not np.mod(i, 2):
        indices_list[i] = np.insert(indices_list[i], 0, idx).tolist()
        #second_indices_list[i] = np.insert(second_indices_list[i], 0, idx).tolist()
        idx += 1
    else:
        indices_list[i] = indices_list[i][0].tolist()
        #second_indices_list[i] = second_indices_list[i][0].tolist()

ordered_tensors_list = [0] * len(tensors_list)
ordered_tensors_name = [0] * len(tensors_list)
#second_ordered_tensors_list = [0] * len(tensors_list)

for i in range(len(ordered_tensors_list)):
    ordered_tensors_list[tensors_list_idx[i]] = cp.copy(tensors_list[i])
    ordered_tensors_name[tensors_list_idx[i]] = cp.copy(tensors_name[i])
    #second_ordered_tensors_list[tensors_list_idx[i]] = cp.copy(np.conj(tensors_list[i]))
'''
for i in range(len(tensors_list)):
    print(ordered_tensors_name[i], ordered_tensors_list[i].shape, ' - ', indices_list[i])
    print(ordered_tensors_name[i], second_ordered_tensors_list[i].shape, ' - ', second_indices_list[i])
'''

final_indices_list = []
final_tensors_list = []
final_tensors_name = []
num = -1
for i in range(len(indices_list)):
    if len(indices_list[i]) == 4:
        indices_list[i][0] = num
        num -= 1
    final_indices_list.append(indices_list[i])
    #final_indices_list.append(second_indices_list[i])
    final_tensors_list.append(ordered_tensors_list[i])
    #final_tensors_list.append(second_ordered_tensors_list[i])
    final_tensors_name.append(ordered_tensors_name[i])
    #final_tensors_name.append(ordered_tensors_name[i] + '*')


for i in range(len(final_tensors_list)):
    print(final_tensors_name[i], final_tensors_list[i].shape, ' - ', final_indices_list[i])

contraction_order = [16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 24, 25, 26, 27, 28, 29, 30, 31, 50, 51, 34, 35, 36, 37, 38, 39, 32, 33, 52, 53, 40, 41, 42, 43, 44, 45, 46, 47, 54, 55]

# Contracting the Tensor Network for normalization
#tn_normalization = tnc.scon(final_tensors_list, tuple(final_indices_list), contraction_order)

# Contracting the Tensor Network with an operator
psi = tnc.scon(final_tensors_list, tuple(final_indices_list), contraction_order)

#operator_expectetion = psi / tn_normalization
#print('\n')
#print('tn_normalization = ', tn_normalization)
#print('psi = ', psi)
#print('operator_expectetion = ', operator_expectetion)
#print('\n')

psi_vec = psi.reshape(1, physical_dim ** n)
normalization = psi_vec.dot(np.transpose(np.conj(psi_vec)))

psi_vec /= np.sqrt(normalization)
density_matrix = np.tensordot(psi_vec, np.transpose(psi_vec), 0)
eigenvals = np.linalg.eigvals(density_matrix)

plt.figure()
plt.scatter(eigenvals)
plt.show()