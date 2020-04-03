import numpy as np
import copy as cp
import SimpleUpdate as su
import StructureMatrixGenerator as smg
import matplotlib.pyplot as plt


# tensor network parameters
N = 4
d = 2
D = 2

# initialize
structureMat, incidenceMat = smg.finitePEPSobcStructureMatrixGenerator(N, N)
tensors, lambdas = smg.randomTensornetGenerator(structureMat, d, D)
n, m = structureMat.shape


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
interactionConstants = [-1] * m
dE = 1e-5
flag = 1
counter = 0
energyPerSite = []
magZ0 = []

# run simple update
tensorsNew, lambdasNew = cp.deepcopy(tensors), cp.deepcopy(lambdas)
while flag:
    energyPerSite.append(np.real(su.energyPerSite(tensorsNew, lambdasNew, structureMat, interactionConstants, 0,
                                                  Opi, Opj, Op_field)))
    magZ0.append(su.singleSiteExpectation(0, tensorsNew, lambdasNew, structureMat, Sz))
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

magnetization = np.zeros((N, N), dtype=float)
arr = np.array(range(N * N))
coords = np.unravel_index(arr, (N, N))
for i, tensor in enumerate(tensorsNew):
    magnetization[coords[0][i], coords[1][i]] = su.singleSiteExpectation(i, tensorsNew, lambdasNew, structureMat, Sz)

plt.figure()
plt.plot(range(counter), energyPerSite, 'o')
plt.show()

plt.figure()
plt.plot(range(counter), magZ0, 'o')
plt.show()

plt.figure()
plt.matshow(magnetization)
plt.colorbar()
plt.show()
