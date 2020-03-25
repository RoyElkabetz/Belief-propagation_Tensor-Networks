import numpy as np

import copy as cp


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

    def add_node(self, alphabet_size, name):
        self.nodes[name] = [alphabet_size, set()]
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
        self.factors[factor_name] = [node_neighbors, tensor]
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
        eigenval = np.eye(alphabet) / alphabet
        #for i in range(alphabet):
        #    eigenval[i, i] = np.random.rand()
        #eigenval /= np.trace(eigenval)
        #unitary = unitary_group.rvs(alphabet)
        #pd = np.matmul(np.transpose(np.conj(unitary)), np.matmul(eigenval, unitary))
        return eigenval

    def make_super_tensor(self, tensor):
        tensor_idx1 = np.array(range(len(tensor.shape)))
        tensor_idx2 = cp.copy(tensor_idx1) + len(tensor_idx1)
        super_tensor_idx_shape = []
        for i in range(len(tensor_idx1)):
            super_tensor_idx_shape.append(tensor_idx1[i])
            super_tensor_idx_shape.append(tensor_idx2[i])
        super_tensor = np.einsum(tensor, tensor_idx1, np.conj(tensor), tensor_idx2, super_tensor_idx_shape)
        return super_tensor

    def sum_product(self, t_max, epsilon):
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
        self.init_save_messages()

        for t in range(t_max):
            old_messages_f2n = factor2node
            old_messages_n2f = node2factor
            for n in nodes.keys():
                alphabet = nodes[n][0]
                for f in nodes[n][1]:
                    neighbor_factors = cp.deepcopy(nodes[n][1])
                    neighbor_factors.remove(f)
                    temp_message = np.ones((alphabet, alphabet), dtype=complex)
                    for item in neighbor_factors:
                        temp_message *= factor2node[item][n]
                    if not neighbor_factors:
                        continue
                    else:
                        node2factor[n][f] = cp.copy(temp_message)
                        node2factor[n][f] /= np.trace(node2factor[n][f])
            for f in factors.keys():
                for n in factors[f][0].keys():
                    factor2node[f][n] = self.f2n_message(f, n, node2factor)
                    '''
                    tensor = cp.deepcopy(factors[f][1])
                    super_tensor = self.make_super_tensor(tensor)
                    neighbor_nodes = cp.deepcopy(factors[f][0].keys())
                    message_idx = [2 * factors[f][0][n], 2 * factors[f][0][n] + 1]
                    neighbor_nodes.remove(n)
                    for item in neighbor_nodes:
                        super_tensor *= self.broadcasting(node2factor[item][f], factors[f][0][item], super_tensor)
                    factor2node[f][n] = np.einsum(super_tensor, range(len(super_tensor.shape)), message_idx)
                    factor2node[f][n] /= np.trace(factor2node[f][n])
                    '''
            self.save_messages(node2factor, factor2node)
        self.messages_n2f = node2factor
        self.messages_f2n = factor2node

    def check_converge(self, n2f_old, f2n_old, n2f_new, f2n_new, epsilon):
        diff = 0
        for n in n2f_old:
            for f in n2f_old[n]:
                diff += np.sum(np.abs(n2f_old[n][f] - n2f_new[n][f]))
        for f in f2n_old:
            for n in f2n_old[f]:
                diff += np.sum(np.abs(f2n_old[f][n] - f2n_new[f][n]))
        if diff < epsilon:
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
            self.node_belief[n] = temp

    def calc_factor_belief(self):
        self.factor_belief = {}
        factors = self.factors
        messages = self.messages_n2f
        keys = factors.keys()
        for f in keys:
            super_tensor = self.make_super_tensor(cp.deepcopy(factors[f][1]))
            neighbors = factors[f][0]
            for n in neighbors.keys():
                super_tensor *= self.broadcasting(messages[n][f], neighbors[n], super_tensor)
            self.factor_belief[f] = super_tensor

    def f2n_message(self, f, n, messages):
        neighbors, tensor = cp.deepcopy(self.factors[f])
        conj_tensor = cp.copy(np.conj(tensor))
        l = cp.copy(len(tensor.shape))
        tensor_idx = range(l)
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
                broadcasting_idx[2 * factors[item][0][object]] = p_dic[object]
                broadcasting_idx[2 * factors[item][0][object] + 1] = p_dic[object + '*']
            p *= self.tensor_broadcasting(f, broadcasting_idx, p)
        p /= np.sum(p)
        return p, p_dic, p_order

    def nodes_marginal(self, p, p_dic, p_order, nodes_list):
        marginal = cp.deepcopy(p)
        final_idx = [0] * len(nodes_list)
        for i in range(len(nodes_list)):
            final_idx[i] = p_dic[nodes_list[i]]
        marginal = np.einsum(marginal, range(len(marginal.shape)), final_idx)
        return marginal




