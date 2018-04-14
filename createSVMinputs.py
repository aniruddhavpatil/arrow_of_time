import numpy as np
import os
X = []
Y = []
file_array = []
for root,dirs,files in os.walk("./train"):
    for name in files:
        # print(name)
        fb = name[7]
        abcd = name[5]
        if (fb == 'F' and (abcd == 'A' or abcd == 'B')) or (fb == 'B' and (abcd == 'C' or abcd == 'D')):
            file_array.append((os.path.join(root, name), 0))
        else:
            file_array.append((os.path.join(root, name), 1))


file_array = sorted(file_array)
print(file_array)

for file,label in file_array:
    hist = np.load(file)
    X.append(hist)
    Y.append(label)
    # if label == 0:
    #     Y.append((1,0))
    # else:
    #     Y.append((0,1))


X = np.asarray(X)
Y = np.asarray(Y)

np.save('./X_train',X)
np.save('./Y_train',Y)

print(X.shape, Y.shape)
