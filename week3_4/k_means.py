from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # use to caculate the distance between 2 points
from sklearn.cluster import KMeans

np.random.seed(20)

means = [[2,2], [8,3], [3,6]]
cov = [[1, 0], [0,1]]
N = 500 # 500 points for each cluster
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3 # the number of clusters

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# a function to display data
def kmeans_display(X, lable):
    K = np.amax(lable) + 1
    X0 = X[lable == 0, :]
    X1 = X[lable == 1, :]
    X2 = X[lable == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8 )
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8 )
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8 )

    plt.axis('equal')
    plt.plot()
    plt.show()

kmeans_display(X, original_label)

# define functions to K-means clustering
# kmeans_init_centers to create initial centers randomly
# kmeans_assign_labels to assign new labels of data for centers
# kmeans_update_centers to update new postion of centers
# has_converged to determined when the algorithms has to stop

# def kmeans_init_centers(X, k):
#     return X[np.random.choice(X.shape[0], k, replace=False)]

# def kmeans_assign_labels(X, centers):
#     D = cdist(X, centers) # caculate the distance between data and centers
#     return np.argmin(D, axis=1)

# def kmeans_update_centers(X, labels, K):
#     centers = np.zeros((K, X.shape[1]))
#     for k in range(K):
#         Xk = X[labels == k, :] #collect all points belongs to k_th cluster
#         centers[k,:] = np.mean(Xk, axis=0) # take average

#     return centers
# def has_converged(centers, new_centers):
#     # return True if 2 sets of centers are the same
#     return(set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

# # k-means clustering algorithms
# def k_means(X, K):
#     centers = [kmeans_init_centers(X, K)]
#     labels = []
#     it = 0
#     while True:
#         labels.append(kmeans_assign_labels(X, centers[-1]))
#         new_centers = kmeans_update_centers(X, labels[-1], K)
#         if has_converged(centers[-1], new_centers):
#             break
#         centers.append(new_centers)
#         it += 1
#     return (centers, labels, it)

# (centers, labebls, it) = k_means(X, K)
# print('Centers founded: ', centers[-1])

# kmeans_display(X, labebls[-1])

# using scikit-learn 
k_means = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers founded: ', k_means.cluster_centers_)
pred_label = k_means.predict(X)
kmeans_display(X, pred_label)