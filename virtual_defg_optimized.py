import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy as cp
from scipy.stats import unitary_group
import time


class Graph:

    def __init__(self, number_of_nodes=None):
        self.node_count = 0
        self.factors = {}
        self.nodes_order = []
        self.node_indices = {}
        self.nodes = {}
        self.node_belief = None
        self.factor_belief = None
        self.factors_count = 0
        self.messages_n2f = None
        self.messages_f2n = None
        self.node_partition = None
        self.factor_partition = None
        self.all_messages = None
        self.rdm_belief = None

    def add_node(self, alphabet_size, name):
        self.nodes[name] = [alphabet_size, set(), self.node_count]
        self.nodes_order.append(name)
        self.node_indices[name] = self.node_count
        self.node_count += 1

    def add_factor(self, node_neighbors, tensor):
        factor_name = 'f' + str(self.factors_count)
        for n in node_neighbors.keys():
            if n not in self.nodes.keys():
                raise IndexError('Tried to factor non exciting node')
            if tensor.shape[node_neighbors[n]] != self.nodes[n][0]:
                raise IndexError('There is a mismatch between node alphabet and tensor size')
            self.nodes[n][1].add(factor_name)
        self.factors[factor_name] = [node_neighbors, tensor, self.factors_count]
        self.factors_count += 1

    def broadcasting(self, message, idx, tensor):
        idx = [2 * idx, 2 * idx + 1]
        new_shape = np.ones(len(tensor.shape), dtype=np.int)
        new_shape[idx] = message.shape
        return np.reshape(message, new_shape)

    def tensor_broadcasting(self, tensor, idx, master_tensor):
        new_shape = np.ones(len(master_tensor.shape), dtype=np.int)
        new_shape[idx] = tensor.shape
        return np.reshape(tensor, new_shape)

    def pd_mat_init(self, alphabet):
        #eigenval = np.eye(alphabet, dtype=complex) / alphabet
        eigenval = np.ones((alphabet, alphabet), dtype=complex)
        #a = np.abs(np.random.rand(alphabet))
        #eigenval = np.zeros((alphabet, alphabet))
        #np.fill_diagonal(eigenval, a / np.sum(a))
        return eigenval


    def make_super_tensor(self, tensor):
        tensor_idx1 = np.array(range(len(tensor.shape)))
        tensor_idx2 = cp.copy(tensor_idx1) + len(tensor_idx1)
        super_tensor_idx_shape = []
        for i in range(1, len(tensor_idx1)):
            super_tensor_idx_shape.append(tensor_idx1[i])
            super_tensor_idx_shape.append(tensor_idx2[i])
        tensor_idx2[0] = tensor_idx1[0]
        super_tensor = np.einsum(tensor, tensor_idx1, np.conj(tensor), tensor_idx2, super_tensor_idx_shape)
        return super_tensor

    def make_super_physical_tensor(self, tensor):
        tensor_idx1 = np.array(range(len(tensor.shape)))
        tensor_idx2 = cp.copy(tensor_idx1) + len(tensor_idx1)
        super_tensor_idx_shape = []
        for i in range(len(tensor_idx1)):
            super_tensor_idx_shape.append(tensor_idx1[i])
            super_tensor_idx_shape.append(tensor_idx2[i])
        #tensor_idx2[0] = tensor_idx1[0]
        super_tensor = np.einsum(tensor, tensor_idx1, np.conj(tensor), tensor_idx2, super_tensor_idx_shape)
        return super_tensor


    def sum_product(self, t_max, epsilon, dumping):
        #print('run BP')
        factors = self.factors
        nodes = self.nodes
        node2factor = {}
        factor2node = {}
        for n in nodes.keys():
            node2factor[n] = {}
            alphabet = nodes[n][0]
            for f in nodes[n][1]:
                node2factor[n][f] = self.pd_mat_init(alphabet)

        for f in factors.keys():
            factor2node[f] = {}
            for n in factors[f][0]:
                alphabet = nodes[n][0]
                factor2node[f][n] = self.pd_mat_init(alphabet)
        #self.init_save_messages()

        for t in range(t_max):
            old_messages_f2n = cp.deepcopy(factor2node)
            old_messages_n2f = cp.deepcopy(node2factor)
            for n in nodes.keys():
                alphabet = nodes[n][0]
                for f in nodes[n][1]:
                    neighbor_factors = cp.copy(nodes[n][1])
                    neighbor_factors.remove(f)
                    temp_message = np.ones((alphabet, alphabet), dtype=complex)
                    for item in neighbor_factors:
                        temp_message *= old_messages_f2n[item][n]
                    #if not neighbor_factors:
                    #    continue
                    #else:
                    node2factor[n][f] = dumping * node2factor[n][f] + (1 - dumping) * temp_message
                    node2factor[n][f] /= np.trace(node2factor[n][f])
            for f in factors.keys():
                for n in factors[f][0].keys():
                    factor2node[f][n] = dumping * factor2node[f][n] + (1 - dumping) * self.f2n_message(f, n, old_messages_n2f)
                    factor2node[f][n] /= np.trace(factor2node[f][n])
            #self.save_messages(node2factor, factor2node)
            self.messages_n2f = node2factor
            self.messages_f2n = factor2node

            if self.check_converge(old_messages_n2f, old_messages_f2n, epsilon):
                break
        print('t_final = ', t)
        #print('\n')

    def check_converge(self, n2f_old, f2n_old, epsilon):
        counter = 0
        num_of_messages = 0
        n2f_new, f2n_new = self.messages_n2f, self.messages_f2n
        for n in n2f_old:
            for f in n2f_old[n]:
                num_of_messages += 1
                if np.sum(np.abs(n2f_old[n][f] - n2f_new[n][f])) < epsilon:
                    counter += 1
        for f in f2n_old:
            for n in f2n_old[f]:
                num_of_messages += 1
                if np.sum(np.abs(f2n_old[f][n] - f2n_new[f][n])) < epsilon:
                    counter += 1
        if counter == num_of_messages:
            return 1
        else:
            return 0

    def save_messages(self, n2f, f2n):
            for n in self.nodes:
                for f in self.nodes[n][1]:
                    self.all_messages[n][f].append(n2f[n][f])
            for f in self.factors:
                for n in self.factors[f][0]:
                    self.all_messages[f][n].append(f2n[f][n])

    def init_save_messages(self):
        self.all_messages = {}
        for n in self.nodes:
            self.all_messages[n] = {}
            for f in self.nodes[n][1]:
                self.all_messages[n][f] = []
        for f in self.factors:
            self.all_messages[f] = {}
            for n in self.factors[f][0]:
                self.all_messages[f][n] = []

    def calc_node_belief(self):
        self.node_belief = {}
        nodes = self.nodes
        messages = self.messages_f2n
        keys = nodes.keys()
        for n in keys:
            alphabet = nodes[n][0]
            temp = np.ones((alphabet, alphabet), dtype=complex)
            for f in nodes[n][1]:
                temp *= messages[f][n]
            self.node_belief[n] = temp / np.trace(temp)

    def calc_rdm_belief(self):
        self.rdm_belief = {}
        factors = self.factors
        messages = self.messages_n2f
        keys = factors.keys()
        for f in keys:
            super_tensor = self.make_super_physical_tensor(factors[f][1])
            neighbors = factors[f][0]
            for n in neighbors.keys():
                super_tensor *= self.broadcasting(messages[n][f], neighbors[n], super_tensor)
            self.rdm_belief[factors[f][2]] = np.einsum(super_tensor, range(len(super_tensor.shape)), [0, 1])
            self.rdm_belief[factors[f][2]] /= np.trace(self.rdm_belief[factors[f][2]])


    def f2n_message(self, f, n, messages):
        neighbors, tensor, index = cp.deepcopy(self.factors[f])
        conj_tensor = cp.copy(np.conj(tensor))
        l = cp.copy(len(tensor.shape))
        tensor_idx = list(range(l))
        for item in neighbors:
            if item == n:
                continue
            message_idx = [self.factors[f][0][item], l + 1]
            final_idx = cp.copy(tensor_idx)
            final_idx[message_idx[0]] = message_idx[1]
            tensor = np.einsum(tensor, tensor_idx, messages[item][f], message_idx, final_idx)
        conj_tensor_idx = cp.copy(tensor_idx)
        conj_tensor_idx[self.factors[f][0][n]] = l + 1
        message_final_idx = [self.factors[f][0][n], l + 1]
        message = np.einsum(tensor, tensor_idx, conj_tensor, conj_tensor_idx, message_final_idx)
        message /= np.trace(message)
        return message

    def exact_joint_probability(self):
        factors = cp.deepcopy(self.factors)
        p_dim = []
        p_order = []
        p_dic = {}
        counter = 0
        for i in range(self.node_count):
            p_dic[self.nodes_order[i]] = counter
            p_dic[self.nodes_order[i] + '*'] = counter + 1
            p_order.append(self.nodes_order[i])
            p_order.append(self.nodes_order[i] + '*')
            p_dim.append(self.nodes[self.nodes_order[i]][0])
            p_dim.append(self.nodes[self.nodes_order[i]][0])
            counter += 2
        p = np.ones(p_dim, dtype=complex)
        for item in factors.keys():
            f = self.make_super_tensor(factors[item][1])
            broadcasting_idx = [0] * len(f.shape)
            for object in factors[item][0]:
                broadcasting_idx[2 * (factors[item][0][object] - 1)] = p_dic[object]
                broadcasting_idx[2 * (factors[item][0][object] - 1) + 1] = p_dic[object + '*']
            permute_tensor_indices = np.argsort(broadcasting_idx)
            f = np.transpose(f, permute_tensor_indices)
            broadcasting_idx = np.sort(broadcasting_idx)
            p *= self.tensor_broadcasting(f, broadcasting_idx, p)
        return p, p_dic, p_order

    def nodes_marginal(self, p, p_dic, p_order, nodes_list):
        marginal = cp.deepcopy(p)
        final_idx = [0] * len(nodes_list)
        for i in range(len(nodes_list)):
            final_idx[i] = p_dic[nodes_list[i]]
        marginal = np.einsum(marginal, range(len(marginal.shape)), final_idx)
        return marginal









