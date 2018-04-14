import numpy as np
import os
from sklearn.decomposition import PCA
Xa = []
Xb = []
Xc = []
Xd = []
Y = []
file_array = []
for root,dirs,files in os.walk("./test"):
    for name in files:
        
        fb = name[7]
        abcd = name[5]
        if (fb == 'F' and (abcd == 'A' or abcd == 'B')) or (fb == 'B' and (abcd == 'C' or abcd == 'D')):
            file_array.append((os.path.join(root, name), 0))
        else:
            file_array.append((os.path.join(root, name), 1))


file_array.sort()
file_array_A = np.asarray(file_array[:60])
file_array_B = np.asarray(file_array[60:120])
file_array_C = np.asarray(file_array[120:180])
file_array_D = np.asarray(file_array[180:])

pca = PCA(n_components=60)

for i in range(60):

    if file_array_A[i][1] == '0':
        Y.append((1,0))
    else:
        Y.append((0,1))

    A = np.load(file_array_A[i][0])
    B = np.load(file_array_B[i][0])
    C = np.load(file_array_C[i][0])
    D = np.load(file_array_D[i][0])

    A = A/A.sum()
    B = B/B.sum()
    C = C/C.sum()
    D = D/D.sum()

    Xa.append(A)
    Xb.append(B)
    Xc.append(C)
    Xd.append(D)

Xa = pca.fit_transform(Xa)
Xb = pca.fit_transform(Xb)
Xc = pca.fit_transform(Xc)
Xd = pca.fit_transform(Xd)

X = np.hstack([Xa,Xb,Xc,Xd])

np.save('./X_train_pca.npy',X)
np.save('./Y_test_pca.npy',Y)
