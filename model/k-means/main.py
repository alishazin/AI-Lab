import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, log=False, max_iters=100):

    np.random.seed(42)

    centers = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):

        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break

        centers = new_centers
        if log:
            plot(X, centers, labels, False, _+1)

    return labels, centers

def plot(X, centers, labels, final=True, i=None):
        
    plt.scatter(X[:, 0], X[:, 1], c=labels, marker='o', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centroids")
    
    if final:
        plt.title("K-means Clustering (Final)")
    else:
        plt.title(f"K-means Clustering (iter={i})")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.show()


X = np.vstack([
    np.random.normal(loc=[2,2], scale=0.5, size=(100,2)),
    np.random.normal(loc=[8,8], scale=0.5, size=(100,2)),
    np.random.normal(loc=[5,2], scale=0.5, size=(100,2)),
])

k = 3
labels, centers = kmeans(X, k, log=True)
plot(X, centers, labels)