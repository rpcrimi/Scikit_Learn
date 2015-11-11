# Classification

While there are many algorithms for classification, we will focus on Support Vector Machines (SVMs) in this section.

## Support Vector Machines
An SVM is a technique to find linear margins between classes of data. To help explain this better, let's take a look at the following diagram:

![SVM](/images/svm.png?raw=true "SVM")

In this example, we have a set of points that have classes of red squares and blue circles. An SVM attempts to find the optimal linear hyperplane that can divide the two classes. The optimal hyperplane is the one that leaves the largest margin between the closest points of opposite classes. These points are called Support Vectors and are critical to SVM's. For example, if another training point fell between the dotted lines in the diagram above, there would be a different optimal hyperplane.

### Non-Linear Hyperplanes
Unfortunately, classes are rarely separable by a linear hyperplane. Most of the time, points lie in a form similar to the diagram below:

![SVM_nonlin](/images/svm_nonlin.gif?raw=true "SVM_nonlin")

To use SVMs on these types of datasets, we must transform the space in which the points live. These transformations are called kernels and ensure that a linear hyperplane to separate the classes will exist. Such kernels can transform the space in the following way:

<img src="/images/svm_kernel.png" width="700"/>

## SVMs in Scikit Learn
Transforming the data's space and calculating the optimal hyperplane can be very hard to do. There are algorithms to handle these challenges, but coding them is very difficult. Luckily, Scikit Learn is here to help. Lets see how Scikit Learn can help classify a dataset that cannot be linearly separable:

1. Create a new python file and import the following modules:
	```
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn import svm
	```

2. Generate some random data and classifications. Our classifications (Y) will be an exclusive or between the random x,y value pairs in X. This will create non-linear boundaries between our classes:
	```
	xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
	np.random.seed(0)
	X = np.random.randn(300, 2)
	Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
	```

3. Let's take a look at our data:
	```
	plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
	plt.xticks(())
	plt.yticks(())
	plt.axis([-3, 3, -3, 3])
	plt.show()
	```
	- After running this script we should see a set of inseparable points similar to the following diagram:
![SVM_nonlin_example](/images/svm_nonlin_example.png?raw=true "SVM_nonlin_example")

4. Without Scikit Learn's help, we would have to transform this space ourselves. However, Scikit Learn's implementation of SVMs will automatically transform the data for us. To do so, add the following code:
	```
	clf = svm.SVC()
	clf.fit(X, Y)
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	```
	- First, we create a new instance of an SVM and assign it to the variable `clf`.
	- Second, we let Scikit Learn fit our data.
	- Third, we assign decision boundaries for each point in the dataset to the list `Z`.
	- Finally, we reshape our list `Z` so we can plot the data.

5. Let's take a look at our data after it has been transformed:
	```
	plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
	contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
	plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
	plt.xticks(())
	plt.yticks(())
	plt.axis([-3, 3, -3, 3])
	plt.show()
	```
	- After running this script we should see the following decision boundaries:
![SVM_nonlin_example_separated](/images/svm_nonlin_example_separated.png?raw=true "SVM_nonlin_example_separated")
	- Scikit Learn transformed our space so that it could classify the data as accurately as possible. However, you should note that this dataset is not classified completely correct. For example, the red points in the center are still classified as blue points. In practice, we can supply kernel values to our SVM for better classification.

The full code for this example can be found [here](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/nonlin_svm.py)

## Other Classification Tools
SVMs are very useful when data can be transformed such that there exists a linear separator. However, there are many cases where we cannot transform our data into such spaces. To learn about other classification tools Scikit Learn provides, visit (http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

Up Next: [Regression](https://github.com/rpcrimi/Scikit_Learn/blob/master/markdown/regression.md)


