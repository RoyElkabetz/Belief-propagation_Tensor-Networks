import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
color_list = ['salmon', 'limegreen', 'mediumturquoise', 'cornflowerblue', 'fuchsia', 'khaki']

import copy as cp
import pickle
import pandas as pd
#from ipywidgets import IntProgress
#from IPython.display import display
import pylustrator
pylustrator.start()


import RandomPEPS as rpeps
import StructureMatrixGenerator as smg
import trivialSimpleUpdate as tsu
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as su
import bmpslib as bmps


file = 'data/data10x10_random_PEPS.npy'
params = 'data/parameters10x10_random_PEPS.npy'
data = np.load(file, allow_pickle=True)
parameters = np.load(params, allow_pickle=True)


# unpack parameters
N, M = parameters[1][1][0], parameters[1][1][1]
bond_dimensions = parameters[2][1]
num_experiments = parameters[10][1]

# unpack data
ATD_D = data[0]
T_BP_D = data[1]
T_SU_D = data[2]

name = '$' + str(N) + ' x ' + str(M) + '$' + ' random PEPS'
plt.figure(figsize=(12, 8))
fonts = 20
names = []
for i, D in enumerate(bond_dimensions):
    plt.scatter(range(1, num_experiments + 1),
                np.asarray(T_BP_D[i]) / np.asarray(T_SU_D[i]),
                color=mcd.CSS4_COLORS[color_list[i]],
                s=50)

    plt.plot(range(1, num_experiments + 1),
             np.mean(np.asarray(T_BP_D[i]) / np.asarray(T_SU_D[i])) * np.ones((num_experiments, 1)),
             '--',
             color=mcd.CSS4_COLORS[color_list[i]])

    names.append('D = ' + str(D))
plt.title(name, fontsize=fonts)
plt.xlabel('Experiment number', fontsize=fonts)
plt.ylabel('$T_{BP}$ / $T_{tSU}$', fontsize=fonts)
plt.xticks(list(range(1, num_experiments + 1, 2)), fontsize=15)
plt.yticks(fontsize=15)
plt.legend(names, fontsize=fonts)
plt.grid()
# plt.savefig('images/' + name + '.svg', format="svg")
# plt.savefig('images/' + name + '.pdf')