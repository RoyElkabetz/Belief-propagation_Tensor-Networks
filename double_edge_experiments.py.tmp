import numpy as np
import DEnFG as fg
import copy as cp
import matplotlib.pyplot as plt
import pylustrator
pylustrator.start()

# parameters
n = 10
alphabet = 2
d = 3
t_max = 100
epsilon = 1e-5

# saving data
node_marginals = np.zeros((alphabet, alphabet, n, t_max))

# generate the graph
g = fg.Graph()

# add physical nodes
for i in range(n):
    g.add_node(alphabet, 'n' + str(i))

# add virtual nodes
for i in range(n):
    g.add_node(d, 'n' + str(g.node_count))

# add factors
for i in range(1, n):
    neighbors = {'n' + str(n + i - 1): 0, 'n' + str(i): 1, 'n' + str(n + i): 2}
    g.add_factor(neighbors, np.random.rand(d, alphabet, d))

# PBC
neighbors = {'n' + str(2 * n - 1): 0, 'n0': 1, 'n' + str(n): 2}
g.add_factor(neighbors, np.random.rand(d, alphabet, d))

# run BP
for t in range(1, t_max):
    g.sum_product(t, epsilon)
    g.calc_node_belief()
    for i in range(n):
        node_marginals[:, :, i, t] = g.node_belief['n' + str(i)]

plt.figure()
plt.plot(list(range(t_max)), node_marginals[0, 0, 0, :], 'o')
plt.plot(list(range(t_max)), node_marginals[0, 1, 0, :], 'v')
plt.plot(list(range(t_max)), node_marginals[1, 0, 0, :], '-')
plt.plot(list(range(t_max)), node_marginals[1, 1, 0, :], '-.')
plt.grid()
#% start: automatic generated code from pylustrator
fig = plt.figure(1)
import matplotlib as mpl
fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}
fig.set_edgecolor("#ffffffff")
fig.set_facecolor("#ffffffff")
fig.axes[0].lines[0].set_markerfacecolor("#a876b4ff")
fig.axes[0].lines[1].set_markeredgecolor("#fc00dbff")
fig.axes[0].lines[2].set_color("#000000ff")
fig.axes[0].lines[2].set_linewidth(2.5)
fig.axes[0].set_position([0.090625, 0.426667, 0.501563, 0.513750])
fig.axes[0].set_xlabel("BP iterations")
fig.axes[0].set_ylabel("value")
fig.axes[0].xaxis.labelpad = 0.320000
fig.axes[0].add_patch(mpl.patches.Rectangle((-4.95, -0.025948182849738734), width=49.5, height=0.2594818284973873))  # id=fig.axes[0].patches[0].new
#% end: automatic generated code from pylustrator
plt.show()




