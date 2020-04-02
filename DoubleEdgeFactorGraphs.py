import numpy as np
import copy as cp
import time


class defg:

    def __init__(self, number_of_nodes=None):
        self.nCounter = 0
        self.factors = {}
        self.nodes_InsertOrder = []
        self.nodes_indices = {}
        self.nodes = {}
        self.nodesBeliefs = None
        self.rdms_dotProduct = None
        self.factorsBeliefs = None
        self.fCounter = 0
        self.messages_n2f = None
        self.messages_f2n = None
        self.node_partition = None
        self.factor_partition = None
        self.all_messages = None
        self.rdms_broadcasting = None

    def add_node(self, alphabet, nName):
        self.nodes[nName] = [alphabet, set(), self.nCounter]
        self.nodes_InsertOrder.append(nName)
        self.nodes_indices[nName] = self.nCounter
        self.nCounter += 1

    def add_factor(self, nodeNeighbors, tensor):
        fName = 'f' + str(self.fCounter)
        for n in nodeNeighbors.keys():
            if n not in self.nodes.keys():
                raise IndexError('Tried to factor non exciting node')
            if tensor.shape[nodeNeighbors[n]] != self.nodes[n][0]:
                raise IndexError('There is a mismatch between node alphabet and tensor index size')
            self.nodes[n][1].add(fName)
        self.factors[fName] = [nodeNeighbors, tensor, self.fCounter]
        self.fCounter += 1

    def messageBroadcasting(self, message, idx, tensor):
        idx = [2 * idx, 2 * idx + 1]
        new_shape = np.ones(len(tensor.shape), dtype=np.int)
        new_shape[idx] = message.shape
        return np.reshape(message, new_shape)

    def messageVBroadcasting(self, message, idx, tensor):
        idx = [2 * (idx - 1), 2 * (idx - 1) + 1]
        new_shape = np.ones(len(tensor.shape), dtype=np.int)
        new_shape[idx] = message.shape
        return np.reshape(message, new_shape)

    def tensorBroadcasting(self, tensor, idx, sizedTensor):
        new_shape = np.ones(len(sizedTensor.shape), dtype=np.int)
        new_shape[idx] = tensor.shape
        return np.reshape(tensor, new_shape)

    def messageInit(self, alphabet):
        return np.ones((alphabet, alphabet), dtype=complex)

    def generateSuperTensor(self, tensor):
        tensorIdx = np.array(range(len(tensor.shape)))
        conjtensorIdx = cp.copy(tensorIdx) + len(tensorIdx)
        superTensorIdx = []
        for i in range(1, len(tensorIdx)):
            superTensorIdx.append(tensorIdx[i])
            superTensorIdx.append(conjtensorIdx[i])
        conjtensorIdx[0] = tensorIdx[0]
        superTensor = np.einsum(tensor, tensorIdx, np.conj(tensor), conjtensorIdx, superTensorIdx)
        return superTensor

    def generateSuperPhysicalTensor(self, tensor):
        tensorIdx = np.array(range(len(tensor.shape)))
        conjtensorIdx = cp.copy(tensorIdx) + len(tensorIdx)
        superTensorIdx = []
        for i in range(len(tensorIdx)):
            superTensorIdx.append(tensorIdx[i])
            superTensorIdx.append(conjtensorIdx[i])
        superTensor = np.einsum(tensor, tensorIdx, np.conj(tensor), conjtensorIdx, superTensorIdx)
        return superTensor

    def sumProduct(self, tmax, epsilon, dumping, initializeMessages=None, printTime=None):
        factors = self.factors
        nodes = self.nodes
        
        # initialize all messages
        if initializeMessages and self.messages_n2f and self.messages_f2n:
            node2factor = self.messages_n2f
            factor2node = self.messages_f2n
        else:
            node2factor = {}
            factor2node = {}
            for n in nodes.keys():
                node2factor[n] = {}
                alphabet = nodes[n][0]
                for f in nodes[n][1]:
                    node2factor[n][f] = self.messageInit(alphabet)
            for f in factors.keys():
                factor2node[f] = {}
                for n in factors[f][0]:
                    alphabet = nodes[n][0]
                    factor2node[f][n] = self.messageInit(alphabet)

        for t in range(tmax):
            # save previous step messages
            preMessages_f2n = cp.deepcopy(factor2node)
            preMessages_n2f = cp.deepcopy(node2factor)
            
            # calculating node to factor (n -> f) messages
            for n in nodes.keys():
                alphabet = nodes[n][0]
                for f in nodes[n][1]:
                    neighbors = cp.deepcopy(nodes[n][1])
                    neighbors.remove(f)
                    tempMessage = np.ones((alphabet, alphabet), dtype=complex)
                    for item in neighbors:
                        tempMessage *= preMessages_f2n[item][n]
                    
                    node2factor[n][f] = dumping * node2factor[n][f] + (1 - dumping) * tempMessage
                    node2factor[n][f] /= np.trace(node2factor[n][f])

            # calculating factor to node (f -> n) messages
            for f in factors.keys():
                for n in factors[f][0].keys():
                    factor2node[f][n] = dumping * factor2node[f][n] + (1 - dumping) * self.f2n_message(f, n, preMessages_n2f)
                    factor2node[f][n] /= np.trace(factor2node[f][n])

            # save this step new messages
            self.messages_n2f = node2factor
            self.messages_f2n = factor2node

            # check if all messages converged
            if self.checkBPconvergence(preMessages_n2f, preMessages_f2n, epsilon):
                break
        if printTime:
            print("BP converged in %d iterations " % t)

    def checkBPconvergence(self, pre_n2f, pre_f2n, epsilon):
        convergenceCounter = 0
        messagesCounter = 0
        n2f_new, f2n_new = self.messages_n2f, self.messages_f2n
        for n in pre_n2f:
            for f in pre_n2f[n]:
                messagesCounter += 1
                if np.sum(np.abs(pre_n2f[n][f] - n2f_new[n][f])) < epsilon:
                    convergenceCounter += 1
        for f in pre_f2n:
            for n in pre_f2n[f]:
                messagesCounter += 1
                if np.sum(np.abs(pre_f2n[f][n] - f2n_new[f][n])) < epsilon:
                    convergenceCounter += 1
        if convergenceCounter == messagesCounter:
            return 1
        else:
            return 0

    def calculateNodesBeliefs(self):
        self.nodesBeliefs = {}
        nodes = self.nodes
        messages = self.messages_f2n
        keys = nodes.keys()
        for n in keys:
            alphabet = nodes[n][0]
            tempMessage = np.ones((alphabet, alphabet), dtype=complex)
            for f in nodes[n][1]:
                tempMessage *= messages[f][n]
            self.nodesBeliefs[n] = tempMessage / np.trace(tempMessage)

    def calculateRDMS_dotProduct(self):
        self.rdms_dotProduct = []
        messages = self.messages_n2f
        for n in range(self.fCounter):
            f = 'f' + str(n)
            neighbors, tensor, index = cp.deepcopy(self.factors[f])
            conjTensor = cp.copy(np.conj(tensor))
            l = len(tensor.shape)
            tensorIdx = list(range(l))
            for node in neighbors:
                messageIdx = [self.factors[f][0][node], l + 1]
                tensorFinalIdx = cp.copy(tensorIdx)
                tensorFinalIdx[messageIdx[0]] = messageIdx[1]
                tensor = np.einsum(tensor, tensorIdx, messages[node][f], messageIdx, tensorFinalIdx)
            conjTensorIdx = cp.copy(tensorIdx)
            conjTensorIdx[0] = l + 1
            messageFinalIdx = [0, l + 1]
            belief = np.einsum(tensor, tensorIdx, conjTensor, conjTensorIdx, messageFinalIdx)
            belief /= np.trace(belief)
            self.rdms_dotProduct.append(belief)

    def calculateRDMS_broadcasting(self):
        self.rdms_broadcasting = {}
        factors = self.factors
        messages = self.messages_n2f
        keys = factors.keys()
        for f in keys:
            super_tensor = self.generateSuperPhysicalTensor(cp.deepcopy(factors[f][1]))
            neighbors = factors[f][0]
            for n in neighbors.keys():
                super_tensor *= self.messageBroadcasting(messages[n][f], neighbors[n], super_tensor)
            idx = list(range(len(super_tensor.shape)))
            self.rdms_broadcasting[factors[f][2]] = np.einsum(super_tensor, idx, [0, 1])
            self.rdms_broadcasting[factors[f][2]] /= np.trace(self.rdms_broadcasting[factors[f][2]])

    def calculateFactorsBeliefs(self):
        self.factorsBeliefs = {}
        factors = self.factors
        messages = self.messages_n2f
        keys = factors.keys()
        for f in keys:
            superTensor = self.generateSuperPhysicalTensor(cp.deepcopy(factors[f][1]))
            neighbors = factors[f][0]
            for n in neighbors.keys():
                superTensor *= self.messageBroadcasting(messages[n][f], neighbors[n], superTensor)
            self.factorsBeliefs[f] = superTensor

    def RDMSfromFactorBeliefs(self):
        rdms = {}
        for i in range(self.fCounter):
            rdms[i] = np.einsum(self.factorsBeliefs['f' + str(i)],
                                list(range(len(self.factorsBeliefs['f' + str(i)].shape))), [0, 1])
            rdms[i] /= np.trace(rdms[i])
        return rdms


    def twoFactorsBelief(self, f1, f2):
        """
        Given names of two factors returns the two factors with the n2f messages absorbed over all edges except the
        common one.
        :param f1: name of factor1, i.e. 'f1'
        :param f2: name of factor2, i.e. 'f24'
        :return: two double-edge factors, where the node 2 factor messages are absorbed over all edges except the common
                 edge btween factor1 and factor2.
        """
        ne1, ten1, idx1 = cp.deepcopy(self.factors[f1])
        ne2, ten2, idx2 = cp.deepcopy(self.factors[f2])
        del_n = []
        for n in ne1:
            if n in ne2:
                del_n.append(n)

        # delete common edge
        for n in del_n:
            del ne1[n]
            del ne2[n]
        messages = self.messages_n2f
        super_tensor1 = self.generateSuperPhysicalTensor(ten1)
        super_tensor2 = self.generateSuperPhysicalTensor(ten2)

        # absorb node2factor messages
        for n in ne1.keys():
            super_tensor1 *= self.messageBroadcasting(messages[n][f1], ne1[n], super_tensor1)
        for n in ne2.keys():
            super_tensor2 *= self.messageBroadcasting(messages[n][f2], ne2[n], super_tensor2)
        return super_tensor1, super_tensor2


    def absorb_message_into_factor_in_env(self, f, nodes_out):
        # return a copy of f super physical tensor with absorbd message from node n
        ne, ten, idx = cp.deepcopy(self.factors[f])
        messages = self.messages_n2f
        super_tensor = self.generateSuperPhysicalTensor(ten)
        for n in ne:
            if n in nodes_out:
                super_tensor *= self.messageBroadcasting(messages[n][f], ne[n], super_tensor)
        return super_tensor


    def absorb_message_into_factor_in_env_efficient(self, f, nodes_out):
        # return a copy of f  tensor with absorbed message from node n
        ne, ten, idx = cp.deepcopy(self.factors[f])
        messages = self.messages_n2f
        for n in ne:
            if n in nodes_out:
                idx = range(len(ten.shape))
                final_idx = range(len(ten.shape))
                final_idx[ne[n]] = len(ten.shape)
                ten = np.einsum(ten, idx, messages[n][f], [len(ten.shape), ne[n]], final_idx)
        return ten


    def f2n_message(self, f, n, messages):
        neighbors, tensor, index = cp.deepcopy(self.factors[f])
        conjTensor = cp.copy(np.conj(tensor))
        l = cp.copy(len(tensor.shape))
        tensorIdx = list(range(l))
        for item in neighbors:
            if item == n:
                continue
            messageIdx = [self.factors[f][0][item], l + 1]
            tensorFinalIdx = cp.copy(tensorIdx)
            tensorFinalIdx[messageIdx[0]] = messageIdx[1]
            tensor = np.einsum(tensor, tensorIdx, messages[item][f], messageIdx, tensorFinalIdx)
        conjTensorIdx = cp.copy(tensorIdx)
        conjTensorIdx[self.factors[f][0][n]] = l + 1
        messageFinalIdx = [self.factors[f][0][n], l + 1]
        message = np.einsum(tensor, tensorIdx, conjTensor, conjTensorIdx, messageFinalIdx)
        message /= np.trace(message)
        return message


    def f2n_message_without_matching_dof_and_broadcasting(self, f, n, messages):
        neighbors, tensor, index = cp.deepcopy(self.factors[f])
        super_tensor = self.generateSuperTensor(tensor)
        for p in neighbors.keys():
            if p == n:
                continue
            super_tensor *= self.messageVBroadcasting(messages[p][f], neighbors[p], super_tensor)
        idx = range(len(super_tensor.shape))
        final_idx = [2 * (neighbors[n] - 1), 2 * (neighbors[n] - 1) + 1]
        message = np.einsum(super_tensor, idx, final_idx)
        return message


    def f2n_message_chnaged_factor_without_matching_dof(self, f, n, messages, new_factor):
        neighbors, tensor, index = cp.deepcopy(self.factors[f])
        super_tensor = self.generateSuperPhysicalTensor(new_factor)
        for p in neighbors.keys():
            if p == n:
                continue
            super_tensor *= self.messageBroadcasting(messages[p][f], neighbors[p], super_tensor)
        idx = range(len(super_tensor.shape))
        idx[0] = idx[1]
        final_idx = [2 * neighbors[n], 2 * neighbors[n] + 1]
        message = np.einsum(super_tensor, idx, final_idx)
        return message



    def f2n_message_chnaged_factor(self, f, n, messages, new_factor):
        neighbors, tensor, index = cp.deepcopy(self.factors[f])
        tensor = new_factor
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
        for i in range(self.nCounter):
            p_dic[self.nodes_InsertOrder[i]] = counter
            p_dic[self.nodes_InsertOrder[i] + '*'] = counter + 1
            p_order.append(self.nodes_InsertOrder[i])
            p_order.append(self.nodes_InsertOrder[i] + '*')
            p_dim.append(self.nodes[self.nodes_InsertOrder[i]][0])
            p_dim.append(self.nodes[self.nodes_InsertOrder[i]][0])
            counter += 2
        p = np.ones(p_dim, dtype=complex)
        for item in factors.keys():
            f = self.generateSuperTensor(factors[item][1])
            broadcasting_idx = [0] * len(f.shape)
            for object in factors[item][0]:
                broadcasting_idx[2 * (factors[item][0][object] - 1)] = p_dic[object]
                broadcasting_idx[2 * (factors[item][0][object] - 1) + 1] = p_dic[object + '*']
            permute_tensor_indices = np.argsort(broadcasting_idx)
            f = np.transpose(f, permute_tensor_indices)
            broadcasting_idx = np.sort(broadcasting_idx)
            p *= self.tensorBroadcasting(f, broadcasting_idx, p)
        return p, p_dic, p_order

    def exact_nodes_marginal(self, p, p_dic, p_order, nodes_list):
        marginal = cp.deepcopy(p)
        final_idx = [0] * len(nodes_list)
        for i in range(len(nodes_list)):
            final_idx[i] = p_dic[nodes_list[i]]
        marginal = np.einsum(marginal, range(len(marginal.shape)), final_idx)
        return marginal


    def special_factor_belief(self, f, legs):
        neighbors, factor, idx = self.factors[f]


        messages = self.messages_n2f

        super_tensor = self.generateSuperPhysicalTensor(cp.deepcopy(factor))
        super_tensor_idx = range(len(super_tensor.shape))
        final_idx = cp.copy(super_tensor_idx)
        for n in neighbors.keys():
            if neighbors[n] in legs:
                continue
            message = messages[n][f]
            message_idx = [2 * neighbors[n], 2 * neighbors[n] + 1]
            for i in message_idx:
                final_idx.remove(i)


    def new_factor_belief(self):
        self.factorsBeliefs = {}
        messages = self.messages_n2f
        keys = self.factors.keys()
        for f in keys:
            neighbors, tensor, index = cp.deepcopy(self.factors[f])
            conj_tensor = cp.copy(np.conj(tensor))
            l = cp.copy(len(tensor.shape))
            tensor_idx = list(range(l))
            for item in neighbors:
                message_idx = [self.factors[f][0][item], l + 1]
                final_idx = cp.copy(tensor_idx)
                final_idx[message_idx[0]] = message_idx[1]
                tensor = np.einsum(tensor, tensor_idx, messages[item][f], message_idx, final_idx)
            conj_tensor_idx = cp.copy(tensor_idx)
            conj_tensor_idx[self.factors[f][0][n]] = l + 1
            message_final_idx = [self.factors[f][0][n], l + 1]
            message = np.einsum(tensor, tensor_idx, conj_tensor, conj_tensor_idx, message_final_idx)
            message /= np.trace(message)









