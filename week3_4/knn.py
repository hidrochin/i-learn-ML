# Example for KNN
# Iris Flower Dataset has 4 dimension:
# length and width of sepal
# length and width of petal
# we will seperate 150 data in Iris Flower Dataset to training set and test set
# Using KNN to predict each datum belongs to what kind of Iris Flower
# This predicted data will be compared with the real flower type of each data in the test set 
# to evaluate the effect of KNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split 

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print('Number of classes: %d' %len(np.unique(iris_Y)))
print('Number of data points: %d' %len(iris_Y))

# X0 = iris_X[iris_Y == 0, :]
# print('\nSample in class 0: \n', X0[:5,:])

# X1 = iris_X[iris_Y == 1, :]
# print('\nSample in class 1: \n', X1[:5,:])

# X2 = iris_X[iris_Y == 2, :]
# print('\nSample in class 1: \n', X2[:5,:])

# pick data train randomly
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=50)

#for example, if each test data we only use 1 nearest training data
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2) # p = 2 means using norm 2 (euclid norm)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Print results for 20 test data points:")
print("Predicted labels: ", Y_pred[20:40])
print("Ground truth    : ", Y_test[20:40])

#evaluation
from sklearn.metrics import accuracy_score
print('Accuracy of 1NN: %.2f %%' %(100*accuracy_score(Y_test, Y_pred)))