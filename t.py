import numpy as np
from scipy.sparse import dok_matrix

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
