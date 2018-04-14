from sklearn import svm
import numpy as np

x_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')

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
