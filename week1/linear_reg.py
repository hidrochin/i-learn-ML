# create information to test the algorithms
# we have a table of height and weight about 15 people
# can we have a prediction about weight through height?
#   height          weight
#   147             49
#   150             50
#   153             51
#   155             52
#   158             54
#   160             56
#   163             58
#   165             59
#   168             60
#   170             72
#   173             63
#   175             64
#   178             66
#   180             67
#   183             68

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

#height
x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight
y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# plt.plot(x, y, 'ro')
# plt.axis([140, 190, 45, 75])
# plt.xlabel("Height (cm)")
# plt.ylabel("Weight (cm)")
# plt.show()

# machine learning compute veirfication

one = np.ones((x.shape[0], 1))
#print(one)
x_bar = np.concatenate((one, x), axis=1)
#print(x_bar)

#caculating weights of the fitting line
A = np.dot(x_bar.T, x_bar) # At . A
b = np.dot(x_bar.T, y) # At . y
w = np.dot(np.linalg.pinv(A), b) # w*
# print('w = ', w)

#prepare fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2, endpoint=True)
y0 = w_0 + w_1*x0

#draw
plt.plot(x.T, y.T, 'ro') # data
plt.plot(x0, y0) # fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

#predict 2 heights which doesnt exist in input table (height = 161, 171)
y1 = w_1*161 + w_0
y2 = w_1*171 + w_0
print('Predict weight with height = 161cm: weigh = %.2f (kg)' %(y1))
print('Predict weight with height = 171cm: weigh = %.2f (kg)' %(y2))

#using scikit-learn to find w_0 and w_1
regression = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regression.fit(x_bar, y)
print( 'Solution found by scikit-learn  : ', regression.coef_ )
print( 'Solution found using ML compute: ', w.T)

#==============================================================
# #ridge regression
# #choose lamda = 1

# #caculating weights of the fitting line
# A = np.dot(x_bar.T, x_bar) # At . A
# b = np.dot(x_bar.T, y) # At . y
# I = np.eye()
# w = np.dot(np.linalg.pinv(A), b) # w*
# # print('w = ', w)

# #prepare fitting line
# w_0 = w[0][0]
# w_1 = w[1][0]
# x0 = np.linspace(145, 185, 2, endpoint=True)
# y0 = w_0 + w_1*x0