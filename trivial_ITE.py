import numpy as np

alphabet = 2
a = np.random.rand(alphabet, alphabet, alphabet, alphabet)
eye = np.kron(np.eye(alphabet), np.eye(alphabet)).reshape([alphabet]*4)
eye = np.transpose(eye, [0, 2, 1, 3])
b = np.einsum('ijkl,jkmn->imnl', a, eye)
c = np.einsum(a, [0, 1, 2, 3], eye, [1, 4, 2, 5], [0, 4, 5, 3])

max_diff = np.abs(a - b).max()
max_diff1 = np.abs(a - c).max()