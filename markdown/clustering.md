# Clustering

Clustering is a form of Unsupervised Learning in which we are trying to find the underlying structure of our data. One form of clustering is K-Means. In this process, we attempt to form K clusters of our data by taking the mean of a cluster after each iteration. We can see an example of this in the figure below:

![K-Means](/images/kmeans_animation.gif?raw=true "K-Means")

Because we define the number of clusters before we start our algorithm, K-Means can be thought of as a form of Unsupervised and Supervised Learning.

## K-Means in Scikit Learn
It is rare to know the number clusters before running a K-Means algorithm. However, Scikit Learn makes it easy to test out different values of K in its implementation of K-Means Clustering.

1. Create a new python file and import the following modules:
	```
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.cluster import KMeans
	from sklearn.datasets import make_blobs
	```

2. Generate random clusters of data:
	```
	n_samples = 10000
	random_state = 170
	X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=5, cluster_std=1.7)
	```
	- Here we use Scikit Learn's make_blobs function to create 5 clusters out of 10,000 points.

3. Let's take a look at our data:
	```
	plt.scatter(X[:, 0], X[:, 1])
	plt.title("Data Points")
	plt.show()
	```
	- After running this script we should see a set of points similar to the following diagram:
![K-Means](/images/kmeans_points.png?raw=true "K-Means")

4. Use Scikit Learn's `KMeans` class to fit our data. While we generated 5 clusters, here we will be naive and tell Scikit Learn to find 3 clusters:
	```
	y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
	```

5. Let's take a look at the 3 clusters Scikit Learn found:
	```
	plt.scatter(X[:, 0], X[:, 1], c=y_pred)
	plt.title("3 Clusters")
	plt.show()
	```
	- After running this script we should see the following 3 clusters:
![K-Means](/images/kmeans_3.png?raw=true "K-Means")

6. Use Scikit Learn to fit our data to 5 clusters:
	```
	y_pred = KMeans(n_clusters=5, random_state=random_state).fit_predict(X)
	```

7. Now, let's take a look at the 5 clusters Scikit Learn found:
	```
	plt.scatter(X[:, 0], X[:, 1], c=y_pred)
	plt.title("5 Clusters")
	plt.show()
	```
	- After running this script we should see the following 5 clusters:
![K-Means](/images/kmeans_5.png?raw=true "K-Means")
	- We now see the 5 clusters correctly grouped by color.

The full code for this example can be found [here](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/cluster.py)

Up Next: [Metrics](https://github.com/rpcrimi/Scikit_Learn/blob/master/markdown/metrics.md)


