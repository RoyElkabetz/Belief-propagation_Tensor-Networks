import numpy as np
import copy as cp

'''
 matrix_product_state is a function which given a tensor coefficients
 of N spins with dimension d wave function 'psi' would convert this tensor
 into a matrix product state (MPS) tensor network (TN) with truncation 
 variable 'k' (the approximation index)

'''


def matrix_product_state(psi, k):
    n = len(psi.shape)
    d = psi.shape[0]
    psi_shape = psi.shape
    if np.sum(np.array(psi_shape) / d) != n:
        raise IndexError('psi has different index sizes')

    u = {}
    s = {}
    v = {}
    mps = {}

    for i in range(n - 1):
        if i == 0:
            new_shape = (d, d ** (n - 1))
            u0, s0, v0 = np.linalg.svd(np.reshape(psi, new_shape), full_matrices=True)

            # Truncation
            trunc = min(k, d)
            if len(s0) > trunc:
                s0[trunc:-1] = 0

            s[i] = np.zeros([u0.shape[1], v0.shape[0]])
            np.fill_diagonal(s[i], s0)
            u[i] = u0
            v_shape = np.ones(n, dtype=int) * d
            v_shape[0] = v_shape[0] ** (n - 1)
            v[i] = np.reshape(v0, v_shape)

        else:
            new_shape = (d ** (n + 1 - i), d ** (n - 1 - i))
            u0, s0, v0 = np.linalg.svd(np.reshape(v[i - 1], new_shape), full_matrices=True)

            # Truncation
            trunc = min(k, d ** (n - i + 1))
            if len(s0) > trunc:
                s0[trunc:-1] = 0

            s[i] = np.zeros([u0.shape[1], v0.shape[0]])
            np.fill_diagonal(s[i], s0)
            u[i] = np.reshape(u0, (d ** (n - i), d, d ** (n + 1 - i)))
            v[i] = v0

    u[i + 1] = v0
    j = 0
    for i in range(0, 2 * n - 2, 2):
        mps[i] = u[j]
        mps[i + 1] = s[j]
        j += 1
    mps[2 * n - 2] = u[n - 1]

    return mps


'''
finding the canonical MPS of psi
'''


def canon_matrix_product_state(psi, k):
    n = len(psi.shape)
    d = psi.shape[0]
    psi_shape = psi.shape
    if np.sum(np.array(psi_shape) / d) != n:
        raise IndexError('psi has different index sizes')

    u = {}
    s = {}
    v = {}
    s_diag = {}
    mps = {}

    for i in range(n - 1):
        if i == 0:
            new_shape = (d, d ** (n - 1))
            u0, s0, v0 = np.linalg.svd(np.reshape(psi, new_shape), full_matrices=True)
            s_diag[i] = s0
            s[i] = np.zeros([u0.shape[1], v0.shape[0]])
            np.fill_diagonal(s[i], s0)
            u[i] = u0
            v[i] = np.matmul(s[i], v0)

        else:
            new_shape = (d ** (i + 1), d ** (n - (i + 1)))
            u0, s0, v0 = np.linalg.svd(np.reshape(v[i - 1], new_shape), full_matrices=True)
            s_diag[i] = s0
            s[i] = np.zeros([u0.shape[1], v0.shape[0]])
            np.fill_diagonal(s[i], s0)
            s_inv_diag = np.zeros(s_diag[i - 1].shape, dtype=float)
            for j in range(len(s_inv_diag)):
                if s_diag[i - 1][j] > 1e-3:
                    s_inv_diag[j] = s_diag[i - 1][j] ** (- 1)
                else:
                    s_inv_diag[j] = 0

            # Truncation of s_inv
            if len(s_inv_diag) > k:
                s_inv_diag[k:] = 0

            u[i] = np.reshape(u0, (d ** i, d ** (i + 2)))
            s_inv = np.zeros((s[i - 1].shape[1], s[i - 1].shape[0]), dtype=float)
            np.fill_diagonal(s_inv, s_inv_diag)

            # Truncation of s
            if len(s_diag[i - 1]) > k:
                s_diag[i - 1][k:] = 0
                s[i - 1] = np.zeros(s[i - 1].shape, dtype=float)
                np.fill_diagonal(s[i - 1], s_diag[i - 1])

            u[i] = np.matmul(s_inv, u[i])
            u[i] = np.reshape(u[i], (s_inv.shape[0], d, d ** (i + 1)))
            v[i] = np.matmul(s[i], v0)
    u[i + 1] = v0
    j = 0
    for i in range(0, 2 * n - 2, 2):
        mps[i] = u[j]
        mps[i + 1] = s[j]
        j += 1
    mps[2 * n - 2] = u[n - 1]

    return mps


def new_canon_matrix_product_state(psi, k):
    n = len(psi.shape)
    d = psi.shape[0]
    psi_shape = psi.shape
    if np.sum(np.array(psi_shape) / d) != n:
        raise IndexError('psi has different index sizes')

    u = {}
    s = {}
    v = {}
    s_diag = {}
    mps = {}

    for i in range(n - 1):
        if i == 0:
            new_shape = (d, d ** (n - 1))
            u0, s0, v0 = np.linalg.svd(np.reshape(psi, new_shape), full_matrices=True)
            s0 = np.trim_zeros(s0)  # could be a problen -  triming zeros but kipping very small values
            u[i], s_diag[i], v[i] = keep_k(u0, s0, v0, k)
            s[i] = np.zeros((s_diag[i].shape[0], s_diag[i].shape[0]))
            np.fill_diagonal(s[i], s0)
            v[i] = np.matmul(s[i], v[i])

        else:
            new_shape = (v[i - 1].shape[0] * d, v[i - 1].shape[1] / d)
            u0, s0, v0 = np.linalg.svd(np.reshape(v[i - 1], new_shape), full_matrices=True)
            s0 = np.trim_zeros(s0)  # could be a problen -  triming zeros but kipping very small values
            u[i], s_diag[i], v[i] = keep_k(u0, s0, v0, k)
            s[i] = np.zeros((s_diag[i].shape[0], s_diag[i].shape[0]))
            np.fill_diagonal(s[i], s0)
            s_inv_diag = s_diag[i] ** (-1)
            u[i] = np.reshape(u[i], (u[i].shape[0] / d, u[i].shape[1] * d))
            s_inv = np.zeros((s[i - 1].shape[1], s[i - 1].shape[0]), dtype=float)
            np.fill_diagonal(s_inv, s_inv_diag)
            u[i] = np.matmul(s_inv, u[i])
            u[i] = np.reshape(u[i], (s_inv.shape[0], d, u[i].shape[1] / d))
            v[i] = np.matmul(s[i], v[i])
    u[i + 1] = v0
    j = 0
    for i in range(0, 2 * n - 2, 2):
        mps[i] = u[j]
        mps[i + 1] = s[j]
        j += 1
    mps[2 * n - 2] = u[n - 1]
    return mps


def keep_k(u, s, v, k):
    l = len(s)
    if l <= k:
        u = u[:, 0:l]
        s = s[0:l]
        v = v[0:l, :]
        return u, s, v
    else:
        u = u[:, 0:k]
        s = s[0:k]
        v = v[0:k, :]
        return u, s, v







