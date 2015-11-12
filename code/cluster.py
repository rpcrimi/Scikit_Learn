import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 10000
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=5, cluster_std=1.7)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Data Points")
plt.show()

# 3 Clusters
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("3 Clusters")
plt.show()

# 5 Clusters
y_pred = KMeans(n_clusters=5, random_state=random_state).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("5 Clusters")
plt.show()