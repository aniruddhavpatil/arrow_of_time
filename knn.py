import cv2
import numpy as np
import matplotlib.pyplot as plt

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(10,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.float32(np.array(range(0,10)))
responses.reshape(10,1)


plt.scatter(trainData[:,0],trainData[:,1],80,'r','^')

newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')


knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 1)
print("data:", trainData)
print("newcomer:", newcomer)
print ("result: ", results[0][0],"\n")
print ("neighbours: ", neighbours,"\n")
print ("distance: ", dist)

# plt.show()
