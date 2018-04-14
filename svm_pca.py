from sklearn import svm
import numpy as np

x_train = np.load('X_train_pca.npy')
y_train = np.load('Y_train_pca_svm.npy')
x_test = np.load('X_test_pca.npy')
y_test = np.load('Y_test_pca_svm.npy')

clf = svm.SVC()
clf.fit(x_train,y_train)

c = 0
score = 0
for i in x_test:
    ans = clf.predict([i])
    if ans == y_test[c]:
        score += 1
    c += 1

print(score,c)
