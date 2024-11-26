import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits import mplot3d

np.random.seed(4)

n_samples = 300
x1 = np.random.randn(n_samples, 1) + np.array([5])
# print(x1)
# x3 = np.random.randn(100, 3) + np.array([3])
x2 = np.random.randn(n_samples, 1) + np.array([10])
# print(x2)

X = np.vstack((x1, x2))
# print(X)

n_components = 3


gmm = GaussianMixture(n_components=n_components)
gmm.fit(X)
labels = gmm.predict(X)
print(gmm.n_iter_)
print(labels)
# labels[-1] = 2

print(type(labels))

# ax = plt.axes(projection ='3d')
# ax.plot3D(X[:, 0], X[:, 1], X[:, 2], 'green')

# plt.scatter(X[:, 0], y=X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', s=50)
plt.scatter(X[:, 0], y=[0] * X.shape[0], c=labels, cmap='viridis', marker='o', edgecolors='k', s=50)
plt.grid()
plt.show()

# print the converged log-likelihood value
print(gmm.lower_bound_)
 
# print the number of iterations needed
# for the log-likelihood value to converge
print(gmm.n_iter_)