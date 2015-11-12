# Linear Regression

Linear Regression is a technique in which semi-linear data is fit to a linear line. We can see an example of Linear Regression in the figure below:

![Linear Regression](/images/linear_regression_example.png?raw=true "Linear Regression")

To calculate this line, we want to minimize the residual sum of squares. This means that the line will be as close as it can be to each point in the dataset.

## SVMs in Scikit Learn

While the math behind Linear Regression is fairly simple, we can use Scikit Learn's implementation to simplify our code.

1. Create a new python file and import the following modules:
	```
	import csv
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn import linear_model
	```
2. Included in the code directory is a file named [cricket.csv](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/cricket.csv)