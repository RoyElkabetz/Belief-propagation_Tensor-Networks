


'''
# ----------------------------------  chain DEnFG n sites any BC-----------------------------------
# parameters

n = 3
alphabet = 2
d = 2

fac1 = np.array([[1., -1.j], [-1.j, 2]])

t_max = 100
epsilon = 1e-5
dumping = 0.2

z = np.array([[1, 0], [0, -1]])
x = np.array([[0, 1], [1, 0]])

# saving data
node_marginals = np.zeros((alphabet, n, t_max), dtype=complex)
exact_node_marginals = np.zeros((alphabet, n, t_max), dtype=complex)
z_measure = np.zeros((n, t_max), dtype=complex)
z_exact = np.zeros((n, t_max), dtype=complex)
x_measure = np.zeros((n, t_max), dtype=complex)
x_exact = np.zeros((n, t_max), dtype=complex)
marginal_error = np.zeros((n, t_max), dtype=complex)


# generate the graph
g = fg.Graph()

# add nodes
for i in range(n):
    g.add_node(alphabet, 'n' + str(i))

# add factors
for i in range(n - 1):
    g.add_factor({'n' + str(i): 0, 'n' + str(i + 1): 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
    #g.add_factor({'n' + str(i): 0, 'n' + str(i + 1): 1}, cp.copy(fac1))

# add PBC
g.add_factor({'n' + str(n - 1): 1, 'n0': 0}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n' + str(n - 1): 0, 'n0': 1}, cp.copy(fac1))

# add loops
#g.add_factor({'n2': 0, 'n4': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n1': 0, 'n4': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n2': 0, 'n5': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n0': 0, 'n3': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n0': 0, 'n2': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n3': 0, 'n5': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))


# exact joint probability
p, p_dic, p_order = g.exact_joint_probability()

# run BP
for t in range(1, t_max):
    g.sum_product(t, epsilon, dumping)
    g.calc_node_belief()
    for i in range(n):
        node = 'n' + str(i)
        node_marginals[:, i, t] = np.linalg.eigvals(g.node_belief[node])
        exact_node_marginals[:, i, t] = np.linalg.eigvals(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*'])))
        z_measure[i, t] = np.trace(np.matmul(g.node_belief[node], z))
        z_exact[i, t] = np.trace(np.matmul(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*'])), z))
        x_measure[i, t] = np.trace(np.matmul(g.node_belief[node], x))
        x_exact[i, t] = np.trace(np.matmul(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*'])), x))
        marginal_error[i, t] = np.sum(np.abs(g.node_belief['n' + str(i)] - g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']))))

#print(np.linalg.eigvals(fac1))
#print('\n')
#print(fac1)

for i in range(n):
    
    plt.figure()
    plt.plot(range(t_max), node_marginals[0, i, :], 'o')
    plt.plot(range(t_max), node_marginals[1, i, :], 'o')
    plt.plot(range(t_max), exact_node_marginals[0, i, :])
    plt.plot(range(t_max), exact_node_marginals[1, i, :])
    plt.title('(n' + str(i) + ',n' + str(i) + '*)')
    plt.legend(['n' + str(i) + '[0]-BP', 'n' + str(i) + '[1]-BP', 'n' + str(i) + '[0]-exact', 'n' + str(i) + '[1]-exact'])
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(range(t_max), z_measure[i, :], 'o')
    plt.plot(range(t_max), x_measure[i, :], 'o')
    plt.plot(range(t_max), z_exact[i, :])
    plt.plot(range(t_max), x_exact[i, :])
    plt.title('z[(n' + str(i) + ',n' + str(i) + '*)]')
    plt.legend(['z_measure', 'x_measure', 'z_exact', 'x_exact'])
    plt.ylim([-1, 1])
    plt.grid()
    plt.show()

legend = []
plt.figure()
plt.title(str(n) + ' factors')
for i in range(n):
    plt.plot(range(t_max), marginal_error[i, :], 'o')
    legend.append('n' + str(i))
plt.ylabel('error')
plt.xlabel('t')
plt.legend(legend)
plt.grid()
plt.show()
'''

'''
# ---------------------------------- 1D DEnFG MPS n sites any BC-----------------------------------
import numpy as np
import DEnFG as fg
import copy as cp
import matplotlib.pyplot as plt
#import pylustrator
#pylustrator.start()

np.random.seed(seed=18)


# parameters

n = 5
alphabet = 2
d = 2


t_max = 30
epsilon = 1e-5
dumping = 0.2

z = np.array([[1, 0], [0, -1]])
x = np.array([[0, 1], [1, 0]])


# generate the graph
g = fg.Graph()

# add nodes
for i in range(n):
    g.add_node(d, 'n' + str(i))

# add factors
for i in range(n - 1):
    g.add_node(alphabet, 'n' + str(g.node_count))
    g.add_factor({'n' + str(i): 1, 'n' + str(i + 1): 2, 'n' + str(g.node_count - 1): 0}, np.random.rand(d, d, alphabet) + 1j * np.random.rand(d, d, alphabet))

# add PBC
g.add_node(alphabet, 'n' + str(g.node_count))
g.add_factor({'n' + str(n - 1): 1, 'n0': 2,  'n' + str(g.node_count - 1): 0}, np.random.rand(d, d, alphabet) + 1j * np.random.rand(d, d, alphabet))

# add loops
#g.add_factor({'n2': 0, 'n4': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n1': 0, 'n4': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n2': 0, 'n5': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n0': 0, 'n3': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n0': 0, 'n2': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
#g.add_factor({'n3': 0, 'n5': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
N = g.node_count
# saving data
node_marginals = np.zeros((alphabet, N, t_max), dtype=complex)
exact_node_marginals = np.zeros((alphabet, N, t_max), dtype=complex)
z_measure = np.zeros((N, t_max), dtype=complex)
z_exact = np.zeros((N, t_max), dtype=complex)
x_measure = np.zeros((N, t_max), dtype=complex)
x_exact = np.zeros((N, t_max), dtype=complex)
marginal_error = np.zeros((N, t_max), dtype=complex)


# exact joint probability
p, p_dic, p_order = g.exact_joint_probability()

# run BP
for t in range(1, t_max):
    g.sum_product(t, epsilon, dumping)
    g.calc_node_belief()
    
    for i in range(N):
        node = 'n' + str(i)
        node_marginals[:, i, t] = np.linalg.eigvals(g.node_belief[node])
        exact_node_marginals[:, i, t] = np.linalg.eigvals(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*'])))
        z_measure[i, t] = np.trace(np.matmul(g.node_belief[node], z))
        z_exact[i, t] = np.trace(np.matmul(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*'])), z))
        x_measure[i, t] = np.trace(np.matmul(g.node_belief[node], x))
        x_exact[i, t] = np.trace(np.matmul(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*'])), x))
        marginal_error[i, t] = np.sum(np.abs(g.node_belief['n' + str(i)] - g.nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(g.nodes_marginal(p, p_dic, p_order, [node, node + '*']))))
    
#print(np.linalg.eigvals(fac1))
#print('\n')
#print(fac1)
for i in range(N):
    plt.figure()
    plt.plot(range(t_max), node_marginals[0, i, :], 'o')
    plt.plot(range(t_max), node_marginals[1, i, :], 'o')
    plt.plot(range(t_max), exact_node_marginals[0, i, :])
    plt.plot(range(t_max), exact_node_marginals[1, i, :])
    plt.title('(n' + str(i) + ',n' + str(i) + '*)')
    plt.legend(
        ['n' + str(i) + '[0]-BP', 'n' + str(i) + '[1]-BP', 'n' + str(i) + '[0]-exact', 'n' + str(i) + '[1]-exact'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(range(t_max), z_measure[i, :], 'o')
    plt.plot(range(t_max), x_measure[i, :], 'o')
    plt.plot(range(t_max), z_exact[i, :])
    plt.plot(range(t_max), x_exact[i, :])
    plt.title('z[(n' + str(i) + ',n' + str(i) + '*)]')
    plt.legend(['z_measure', 'x_measure', 'z_exact', 'x_exact'])
    plt.ylim([-1, 1])
    plt.grid()
    plt.show()

legend = []
plt.figure()
plt.title(str(n) + ' factors')
for i in range(N):
    plt.plot(range(t_max), marginal_error[i, :], 'o')
    legend.append('n' + str(i))
plt.ylabel('error')
plt.xlabel('t')
plt.legend(legend)
plt.grid()
plt.show()

#------------------------------------------here
'''
'''
import numpy as np
import copy as cp
import BPupdate as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import ncon
import DEnFG as fg


#---------------------- Tensor Network paramas ------------------

N = 4 # number of spins
L = np.int(np.sqrt(N))

t_max = 50
epsilon = 1e-5
dumping = 0.2

d = 2  # virtual bond dimension
p = 2  # physical bond dimension
D_max = 2  # maximal virtual bond dimension



#------------- generating the finite PEPS structure matrix------------------
imat = np.zeros((N, 2 * N), dtype=int)
smat = np.zeros((N, 2 * N), dtype=int)
n, m = imat.shape
for i in range(n):
    imat[i, 2 * i] = 1
    imat[i, 2 * i + 1] = 1
    imat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 1
    imat[i, 2 * np.mod(i + L, N) + 1] = 1

    smat[i, 2 * i] = 1
    smat[i, 2 * i + 1] = 2
    smat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 3
    smat[i, 2 * np.mod(i + L, N) + 1] = 4


TT = []
for ii in range(n):
    TT.append(np.arange(np.float(p * (d ** 4))).reshape(p, d, d, d, d))
LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)

graph = fg.Graph()
graph = su.PEPStoDEnFG_transform(graph, cp.deepcopy(TT), cp.deepcopy(LL), smat)


BP_in_time = np.zeros((p, p, N, t_max))
exact_marg = np.zeros((p, p, N))
p, p_dic, p_order = graph.exact_joint_probability()

for t in range(1, t_max):

    graph.sum_product(t, epsilon, dumping)
    graph.calc_node_belief()
    #for n in range(N):
    #    BP_in_time[:, :, n, t] = graph.node_belief['n' + str(len(LL) + n)]

plt.figure()
for n in range(N):
    plt.plot(range(t_max), BP_in_time[0, 0, n, :], '^')
    plt.plot(range(t_max), BP_in_time[0, 1, n, :], '^')
    plt.plot(range(t_max), BP_in_time[1, 0, n, :], '^')
    plt.plot(range(t_max), BP_in_time[1, 1, n, :], '^')
plt.grid()
plt.show()
'''

# ---------------------------------- 1D DEnFG MPS n sites any BC-----------------------------------
import numpy as np
import DoubleEdgeFactorGraphs as fg
import copy as cp
import matplotlib.pyplot as plt
#from glassy_1D_AFH_chain import TT
#from glassy_1D_AFH_chain import LL

# import pylustrator
# pylustrator.start()

np.random.seed(seed=18)

# parameters

n = 3
alphabet = 2
d = 2

t_max = 20
epsilon = 1e-5
dumping = 0.2

z = np.array([[1, 0], [0, -1]])
x = np.array([[0, 1], [1, 0]])

# generate the graph
g = fg.defg()

# add nodes
for i in range(n):
    g.add_node(d, 'n' + str(i))

# add factors
for i in range(n - 1):
    g.add_factor({'n' + str(i): 1, 'n' + str(i + 1): 2}, np.random.rand(d, d, alphabet) + 1j * np.random.rand(d, d, alphabet))

# add PBC
#g.add_factor({'n' + str(n - 1): 1, 'n0': 2}, np.random.rand(d, d, alphabet) + 1j * np.random.rand(d, d, alphabet))

# add loops
# g.add_factor({'n2': 0, 'n4': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
# g.add_factor({'n1': 0, 'n4': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
# g.add_factor({'n2': 0, 'n5': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
# g.add_factor({'n0': 0, 'n3': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
# g.add_factor({'n0': 0, 'n2': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
# g.add_factor({'n3': 0, 'n5': 1}, np.random.rand(d, d) + 1j * np.random.rand(d, d))
N = g.nCounter
# saving data
node_marginals = np.zeros((alphabet, N, t_max), dtype=complex)
exact_node_marginals = np.zeros((alphabet, N, t_max), dtype=complex)
z_measure = np.zeros((N, t_max), dtype=complex)
z_exact = np.zeros((N, t_max), dtype=complex)
x_measure = np.zeros((N, t_max), dtype=complex)
x_exact = np.zeros((N, t_max), dtype=complex)
marginal_error = np.zeros((N, t_max), dtype=complex)

# exact joint probability
p, p_dic, p_order = g.exact_joint_probability()

# run BP
for t in range(1, t_max):
    g.sumProduct(t, epsilon, dumping)
    g.calculateNodesBeliefs()

    for i in range(N):
        node = 'n' + str(i)
        node_marginals[:, i, t] = np.linalg.eigvals(g.nodesBeliefs[node])
        exact_node_marginals[:, i, t] = np.linalg.eigvals(
            g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(
                g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*'])))
        z_measure[i, t] = np.trace(np.matmul(g.nodesBeliefs[node], z))
        z_exact[i, t] = np.trace(np.matmul(g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(
            g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*'])), z))
        x_measure[i, t] = np.trace(np.matmul(g.nodesBeliefs[node], x))
        x_exact[i, t] = np.trace(np.matmul(g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(
            g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*'])), x))
        marginal_error[i, t] = np.sum(np.abs(
            g.nodesBeliefs['n' + str(i)] - g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*']) / np.trace(
                g.exact_nodes_marginal(p, p_dic, p_order, [node, node + '*']))))

# print(np.linalg.eigvals(fac1))
# print('\n')
# print(fac1)
for i in range(N):
    plt.figure()
    plt.plot(range(t_max), node_marginals[0, i, :], 'o')
    plt.plot(range(t_max), node_marginals[1, i, :], 'o')
    plt.plot(range(t_max), exact_node_marginals[0, i, :])
    plt.plot(range(t_max), exact_node_marginals[1, i, :])
    plt.title('(n' + str(i) + ',n' + str(i) + '*)')
    plt.legend(
        ['n' + str(i) + '[0]-BP', 'n' + str(i) + '[1]-BP', 'n' + str(i) + '[0]-exact', 'n' + str(i) + '[1]-exact'])
    plt.ylim([0, 1])
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(range(t_max), z_measure[i, :], 'o')
    plt.plot(range(t_max), x_measure[i, :], 'o')
    plt.plot(range(t_max), z_exact[i, :])
    plt.plot(range(t_max), x_exact[i, :])
    plt.title('z[(n' + str(i) + ',n' + str(i) + '*)]')
    plt.legend(['z_measure', 'x_measure', 'z_exact', 'x_exact'])
    plt.ylim([-1, 1])
    plt.grid()
    plt.show()

legend = []
plt.figure()
plt.title(str(n) + ' factors')
for i in range(N):
    plt.plot(range(t_max), marginal_error[i, :], 'o')
    legend.append('n' + str(i))
plt.ylabel('error')
plt.xlabel('t')
plt.legend(legend)
plt.grid()
plt.show()

# ------------------------------------------here
