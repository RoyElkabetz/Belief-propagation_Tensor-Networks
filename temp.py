from numba import jit
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import StructureMatrixGenerator as tnf
import DoubleEdgeFactorGraphs as defg
import SimpleUpdate as BP
import pickle
import pandas as pd


'''
Ek = 15
n, m = 4, 4
env_size = 1
smat, imat = tnf.PEPS_OBC_smat_imat(n, m)
emat = tnf.PEPS_OBC_edge_rect_env(Ek, smat, [n, m], env_size)
inside, outside = tnf.PEPS_OBC_divide_edge_regions(emat, smat)
omat = np.where(emat > -1, np.arange(n * m).reshape(n, m), -1)
tensors = omat[np.nonzero(emat > -1)]
sub_omat = tnf.PEPS_OBC_edge_environment_sub_order_matrix(emat)
n, m = sub_omat.shape
if n < m:
    a = np.transpose(sub_omat)
'''
'''
p, d = 2, 3
n, m = 3, 2
smat, imat = tnf.PEPS_OBC_smat_imat(n, m)
TT, LL = tnf.PEPS_OBC_random_tn_gen(smat, p, d)
TT_new = BP.absorb_all_bond_vectors(TT, LL, smat)
TT_prime = tnf.PEPS_OBC_broadcast_to_Itai(TT_new, [n, m], p, d)
for t, T in enumerate(TT_prime):
    print(np.max(np.abs(TT_prime[t] - TT_prime_tilde[t])))
'''
'''
def quicksort(arr, lo, hi):
    if lo < hi:
        pi = partition(arr, lo, hi)
        quicksort(arr, lo, pi - 1)
        quicksort(arr, pi + 1, hi)

def partition(arr, lo, hi):
    i = lo - 1
    p = arr[hi]
    for j in range(lo, hi):
        if arr[j] < p:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1
'''
'''
a = [6, 5, 77, 34, 2, 91, 8, 4, 1, 4, 0, 2, 45]
quicksort(a, 0, len(a) - 1)
print(a)
'''

'''
matrix = np.arange(16).reshape(8, 2)
order = []
n, m = matrix.shape
row = 0
col = 0

d1 = 0
d2 = 0
flag = -1
while row < n - 1 or col < m - 1:
    order.append(matrix[row, col])

    if flag == 1:
        if row < (n - 1):
            row += 1
        else:
            col += 1
        if row == n - 1 and col > 0:
            r = min(n, m)
        else: r = row - col
        for k in range(r):
            order.append(matrix[row, col])
            if col < (m - 1):
                col += 1
                row -= 1
    if flag == -1:
        if col < m - 1:
            col += 1
        else:
            row += 1
        if col == m - 1 and row > 0:
            r = min(m, n)
        else: r = col - row
        for k in range(r):
            order.append(matrix[row, col])
            if row < (n - 1):
                row += 1
                col -= 1
    flag *= -1
order.append(matrix[row, col])
'''
'''
matrix = np.arange(18).reshape(6, 3)

N, M = matrix.shape

row, column = 0, 0

direction = 1

result = []

while row < N and column < M:
    result.append(matrix[row][column])
    new_row = row + (-1 if direction == 1 else 1)
    new_column = column + (1 if direction == 1 else -1)

    if new_row < 0 or new_row == N or new_column < 0 or new_column == M:

        if direction:
            row += (column == M - 1)
            column += (column < M - 1)
        else:
            column += (row == N - 1)
            row += (row < N - 1)

        direction = 1 - direction
    else:
        row = new_row
        column = new_column

'''
'''
def quicksort2(arr, lo, hi):
    if lo < hi:
        pi = partition2(arr, lo, hi)
        quicksort2(arr, lo, pi - 1)
        quicksort2(arr, pi + 1, hi)

def partition2(arr, lo, hi):
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1
'''
'''
numRows = 5
pascal = []
for row in range(numRows):
    pascal.append([1] * (row + 1))

if numRows > 2:
    for row in range(2, numRows):
        for i in range(1, len(pascal[row]) - 1):
            pascal[row][i] = pascal[row - 1][i - 1] + pascal[row - 1][i]
            '''

'''
def strStr(haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    n = len(needle)
    if n == 0:
        return 0
    if n > len(haystack):
        return -1
    counter = 0
    for i in range(len(haystack) - n + 1):
        if haystack[i] == needle[0]:
            a = haystack[i: n + i]
            for j in range(len(a)):
                if a[j] == needle[j]:
                    counter += 1
            if counter == n:
                return i
            else:
                counter = 0
    return -1

print(strStr("hello", "ll"))
'''

'''
nums = [0,0,1,1,1,2,3,3,4, 5]
lenght = 1
dups = 0
for i in range(1, len(nums)):
    while nums[i - 1] == nums[i + dups]:
        dups += 1
        if i + dups == len(nums):
            break
    if i + dups == len(nums):
        break
    if dups > 0:
        nums[i] = nums[i + dups]
        lenght += 1
        dups -= 1
print(nums[:lenght])
'''
'''
###################### Sorting Algorithms #######################

def minArgMin(s):
    idx = 0
    if not s:
        return None
    if len(s) == 1:
        return idx, s[idx]
    minimum = s[idx]
    for i in range(len(s)):
        if s[i] < minimum:
            minimum = s[i]
            idx = i
    return idx, minimum


def selectionSort(s):
    if not s:
        return s
    i = 0
    n = len(s)
    while i < n - 1:
        idx, _ = minArgMin(s[i:])
        s[i], s[i + idx] = s[i + idx], s[i]
        i += 1


def bubbleSort(s):
    for i in range(len(s) - 1):
        for j in range(len(s) - 1):
            if s[j] > s[j + 1]:
                s[j], s[j + 1] = s[j + 1], s[j]


def recursiveBubbleSort(s):
    n = len(s)
    for i in range(n):
        if n - i == 1:
            return
        for j in range(n - 1):
            if s[j] > s[j + 1]:
                s[j], s[j + 1] = s[j + 1], s[j]
        recursiveBubbleSort(s[0:n - 1 - i])


def partition(s, low, high):
    i = low - 1
    pivot = s[high]
    for j in range(low, high):
        if s[j] < pivot:
            i += 1
            s[i], s[j] = s[j],  s[i]
    s[i + 1], s[high] = s[high], s[i + 1]
    return i + 1


def quickSort(s, low, high):
    if low < high:
        p = partition(s, low, high)
        quickSort(s, low, p - 1)
        quickSort(s, p + 1, high)


def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Finding the mid of the array
        L = arr[:mid]  # Dividing the array elements
        R = arr[mid:]  # into 2 halves

        mergeSort(L)  # Sorting the first half
        mergeSort(R)  # Sorting the second half

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


a = [9, 4, 5, 3, 8, 0, 4, 5, 12, 2, 18, 1]
mergeSort(a)
'''

'''
################################### random PEPS experiments ###################################
import RandomPEPS_main as rpm

n = 100

BP_times = np.zeros((1, n), dtype=float)
SU_times = np.zeros((1, n), dtype=float)
ttd_su_su0_bmps_stat = np.zeros((1, n), dtype=float)
ttd_su_su0_stat = np.zeros((1, n), dtype=float)
ttd_bp_su0_bmps_stat = np.zeros((1, n), dtype=float)
ttd_bp_su0_stat = np.zeros((1, n), dtype=float)
ttd_su_bp_stat = np.zeros((1, n), dtype=float)
ttd_su_su_bmps_stat = np.zeros((1, n), dtype=float)
ttd_bp_su_bmps_stat = np.zeros((1, n), dtype=float)
ttd_su0_su0_bmps_stat = np.zeros((1, n), dtype=float)
for i in range(n):
    print('\n')
    print('i:', i)
    time_bp, time_su, ttd_su_su0, ttd_su_bp, ttd_bp_su0, ttd_su_su_bmps, ttd_su_su0_bmps, ttd_su0_su0_bmps, ttd_bp_su_bmps, ttd_bp_su0_bmps = rpm.randomPEPSmainFunction()
    BP_times[0][i] = time_bp
    SU_times[0][i] = time_su
    ttd_su_su0_bmps_stat[0][i] = ttd_su_su0_bmps[0]
    ttd_su_su0_stat[0][i] = ttd_su_su0[0]
    ttd_bp_su0_bmps_stat[0][i] = ttd_bp_su0_bmps[0]
    ttd_bp_su0_stat[0][i] = ttd_bp_su0[0]
    ttd_su_bp_stat[0][i] = ttd_su_bp[0]
    ttd_su_su_bmps_stat[0][i] = ttd_su_su_bmps[0]
    ttd_bp_su_bmps_stat[0][i] = ttd_bp_su_bmps[0]
    ttd_su0_su0_bmps_stat[0][i] = ttd_su0_su0_bmps[0]
BP_av_time = np.sum(BP_times) / n
SU_av_time = np.sum(SU_times) / n
ttd_su_su0_bmps_stat_av = np.sum(ttd_su_su0_bmps_stat) / n
ttd_su_su0_stat_av = np.sum(ttd_su_su0_stat) / n
ttd_bp_su0_bmps_stat_av = np.sum(ttd_bp_su0_bmps_stat) / n
ttd_bp_su0_stat_av = np.sum(ttd_bp_su0_stat) / n
ttd_su_bp_stat_av = np.sum(ttd_su_bp_stat) / n
ttd_su_su_bmps_stat_av = np.sum(ttd_su_su_bmps_stat) / n
ttd_bp_su_bmps_stat_av = np.sum(ttd_bp_su_bmps_stat) / n
ttd_su0_su0_bmps_stat = np.sum(ttd_su0_su0_bmps_stat) / n

print('------------ Total Trace Distance (single site) statistics ------------')
print('SU - SU0 : ', ttd_su_su0_stat_av)
print('SU - BP : ', ttd_su_bp_stat_av)
print('BP - SU0 : ', ttd_bp_su0_stat_av)
print('SU - SU_bmps : ', ttd_su_su_bmps_stat_av)
print('SU - SU0_bmps : ', ttd_su_su0_bmps_stat_av)
print('SU0 - SU0_bmps : ', ttd_su0_su0_bmps_stat)
print('BP - SU_bmps : ', ttd_bp_su_bmps_stat_av)
print('BP - SU0_bmps : ', ttd_bp_su0_bmps_stat_av)
print('------------------------------------------------------------')
print('\n')

data = [BP_av_time, SU_av_time, ttd_bp_su0_bmps_stat_av, ttd_su_su0_stat_av, ttd_bp_su0_bmps_stat_av, ttd_bp_su0_stat_av, ttd_su_bp_stat_av, ttd_su_su_bmps_stat_av, ttd_bp_su_bmps_stat_av, ttd_su0_su0_bmps_stat]
file_name = "2019_02_27_1_100_OBC_Random_PEPS"
pickle.dump(data, open(file_name + '_D_3_all_data.p', "wb"))
'''

import RandomPEPS_main as rpm

n = 100
ttd_SU_bMPO = np.zeros((1, n), dtype=float)
ttd_SU_BP = np.zeros((1, n), dtype=float)
ttd_bMPO16_bMPO32 = np.zeros((1, n), dtype=float)
#ttd_bMPO8_bMPO16, av_ttd_bMPO8_bMPO16 = pickle.load(open("2DTN_data/2019_02_27_2_100_OBC_Random_PEPS_D_3_SU0_8vs16.p", "rb"))

for i in range(n):
    print('\n')
    print('i:', i)
    rho_SU_0_bmps_single, rho_SU, rho_BP = rpm.randomPEPSmainFunction()
    rho_SU_0_bmps16_single = rho_SU_0_bmps_single[0]
    rho_SU_0_bmps32_single = rho_SU_0_bmps_single[1]
    for j in range(n):
        ttd_SU_bMPO[0][i] += BP.trace_distance(rho_SU_0_bmps16_single[j], rho_SU[j]) / n
        ttd_SU_BP[0][i] += BP.trace_distance(rho_BP[j], rho_SU[j]) / n
        ttd_bMPO16_bMPO32[0][i] += BP.trace_distance(rho_SU_0_bmps16_single[j], rho_SU_0_bmps32_single[j]) / n
av_ttd_SU_bMPO = np.sum(ttd_SU_bMPO) / n
av_ttd_SU_BP = np.sum(ttd_SU_BP) / n
av_ttd_bMPO16_bMPO32 = np.sum(ttd_bMPO16_bMPO32) / n
print('\n')
print('av_ttd_SU_bMPO', av_ttd_SU_bMPO)
print('av_ttd_SU_BP', av_ttd_SU_BP)
file_name1 = "2020_03_04_1_100_OBC_Random_PEPS_SU_bMPO16"
file_name2 = "2020_03_04_1_100_OBC_Random_PEPS_SU_BP"
file_name3 = "2020_03_04_1_100_OBC_Random_PEPS_bMPO16_bMPO32"
a1 = [ttd_SU_bMPO, av_ttd_SU_bMPO]
a2 = [ttd_SU_BP, av_ttd_SU_BP]
a3 = [ttd_bMPO16_bMPO32, av_ttd_bMPO16_bMPO32]

pickle.dump(a1, open(file_name1 + '.p', "wb"))
pickle.dump(a2, open(file_name2 + '.p', "wb"))
pickle.dump(a3, open(file_name3 + '.p', "wb"))


data = np.zeros((100, 3), dtype=float)
data[:, 0] = ttd_SU_BP
data[:, 1] = ttd_SU_bMPO
data[:, 2] = ttd_bMPO16_bMPO32
# [BP energy, SU energy]
df = pd.DataFrame(data, columns=['SU-BP', 'SU-bMPO(16)', 'bMPO(16)-bMPO(32)'], index=range(1, 101))
filepath = '2020_03_04_trivial-SU_BP_experiment' + '.xlsx'
df.to_excel(filepath, index=True)

