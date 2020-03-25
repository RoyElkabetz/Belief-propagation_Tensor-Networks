import numpy as np
import copy as cp
import SimpleUpdate as su
import StructureMatrixGenerator as smg
import matplotlib.pyplot as plt


# tensor network parameters
N = 10
d = 2
D = 3

# initialize
structureMat = smg.finitePEPSobcStructureMatrixGenerator(N, N)
tensors, lambdas = smg.randomTensornetGenerator(structureMat, d, D)
N, M = structureMat.shape


# imaginary time evolution parameters
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
X = np.array([[0, 1], [1, 0]])
Sz = 0.5 * Z
Sy = 0.5 * Y
Sx = 0.5 * X
Opi = [Sx, Sy, Sz]
Opj = [Sx, Sy, Sz]
Op_field = np.eye(d)
timeStep = 0.1
interactionConstants = [1] * M
dE = 1e-5
flag = 1
counter = 0
energyPerSite = []

# run simple update
tensorsNew, lambdasNew = cp.deepcopy(tensors), cp.deepcopy(lambdas)
while flag:
    energyPerSite.append(np.real(su.energy_per_site(tensorsNew, lambdasNew, structureMat, interactionConstants, 0,
                                                          Opi, Opj, Op_field)))
    tensorsNew, lambdasNew = su.simpleUpdate(tensors=tensorsNew,
                                               weights=lambdasNew,
                                               timeStep=timeStep,
                                               interactionConst=interactionConstants,
                                               fieldConst=0,
                                               iOperators=Opi,
                                               jOperators=Opj,
                                               fieldOperators=Op_field,
                                               smat=structureMat,
                                               Dmax=D,
                                               type='SU',
                                               )
    if counter >= 1:
        if np.abs(energyPerSite[counter] - energyPerSite[counter - 1]) <= dE:
            flag = 0
    counter += 1

plt.figure()
plt.plot(range(counter), energyPerSite, 'o')
plt.show()
