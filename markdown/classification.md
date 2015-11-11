# Classification

While there are many algorithms for classification, we will focus on Support Vector Machines (SVMs) in this section.

## Support Vector Machines
An SVM is a technique to find linear margins between classes of data. To help explain this better, let's take a look at the following diagram:

![SVM](/images/svm.png?raw=true "SVM")

In this example, we have a set of points that have classes of red squares and blue circles. An SVM attempts to find the optimal linear hyperplane that can divide the two classes. The optimal hyperplane is the one that leaves the largest margin between the closest points of opposite classes. These points are called Support Vectors and are critical to SVM's. For example, if another training point fell between the dotted lines in the diagram above, there would be a different optimal hyperplane.

### Non-Linear Hyperplanes
Unfortunately, classes are rarely separable by a linear hyperplane. Most of the time, points lie in a form similar to the diagram below:

![SVM_nonlin](/images/svm_nonlin.gif?raw=true "SVM_nonlin")

To use SVMs on these types of datasets, we must transform the space in which the points live. These transformations are called kernels. Such kernels can transform the space in the following way:

<img src="/images/svm_kernel.png" width="700"/>