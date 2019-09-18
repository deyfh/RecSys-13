import numpy as np
from scipy.sparse import dok_matrix
import scipy.sparse as sparse
from collections import defaultdict

from lib.TimeAwareMF import TimeAwareMF
from lib.metrics import precisionk, recallk


'''
S = dok_matrix((5, 5), dtype=np.float32)
for i in range(5):
    for j in range(5):
        S[i, j] = i + j
print(S)
print(S.toarray())
print(np.shape(S))

A = [S for _ in range(10)]
print(A)
print(np.size(A))
print(np.shape(A))
'''
'''''
visited_lids = defaultdict(set)
visited_lids[5].add(5)
visited_lids[5].add(6)
visited_lids[6].add(5)
# visited_lids[3].add('aaa')
print(visited_lids)
print(np.size(visited_lids))
print(np.shape(visited_lids))

sparse_training_matrices = [sparse.dok_matrix((10, 10)) for _ in range(5)]

sparse_training_matrices[2][2, 3] = 1.0 / (1.0 + 1.0 / 2)
sparse_training_matrices[2][2, 4] = 1.0 / (1.0 + 1.0 / 2)
print(sparse_training_matrices)
print(sparse_training_matrices[2][2, 3])
'''


sigma = ([[0, 0, 3], [4, 0, 6], [7, 8, 9]])
a = [sparse.dia_matrix(sigma_t) for sigma_t in sigma]
b = sparse.coo_matrix(sigma)
# print(a(1,2))
# print(b)


row = np.array([0, 3, 1, 0, 0])
col = np.array([0, 3, 1, 2, 0])
data = np.array([4, 5, 7, 9, 0])
c = sparse.coo_matrix((data, (row, col)), shape=(4, 4)).todense()
d = sparse.coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
# print("-----------------------------------------------")
# print(c.shape)
#print("-----------------------------------------------")
c = dok_matrix((4, 4))
c[0, 0] = 1
c[1, 1] = 2
c[2, 0] = 3
c[2, 3] = 4
d = c.tocsr()
#print(c)
print("-----------------------------------------------")

sigmaa = [np.zeros(10) for _ in range(6)]
for _ in range(6):
    sigmaa[_][4] = 1
sigmaa = [sparse.dia_matrix(sigma_t) for sigma_t in sigma]
# sigmaa[5][5] = 55

print(sigmaa[2])

