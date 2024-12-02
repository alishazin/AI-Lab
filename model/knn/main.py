import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate Bunch of Points and divide it as train and test set
X, y = make_classification(n_samples=300, n_classes=2, n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=19)

# Initialize the Model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train and Test the Model

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy Score: ", ac)

# Plot the Points in 2D

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', alpha=0.8)

plt.title(f"K-Nearest Neighbours (k={k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="y Value")
plt.grid()
plt.show()

# Plot the Points in 3D
 
ax = plt.axes(projection ='3d') 
ax.scatter(X[:, 0], X[:, 1], y, c=y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Plot the Contour

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
Z = knn.predict(np.c_[x1x1.ravel(), x2x2.ravel()])
Z = Z.reshape(x1x1.shape)

plt.contourf(x1x1, x2x2, Z)

plt.title(f"K-Nearest Neighbours (k={k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="y Value")
plt.grid()
plt.show()