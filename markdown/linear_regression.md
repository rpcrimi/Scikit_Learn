# Linear Regression

Linear Regression is a technique in which the best-fit line is calculated for semi-linear data. The line can be used to predict results based on different input values. The figure below shows an example of Linear Regression:

![Linear Regression](/images/linear_regression_example.jpg?raw=true "Linear Regression")

To determine the regression line from the data points, we use the technique of minimizing the residual sum of squares. The resulting line will be as close to all of the data points as possible.

## SVMs in Scikit Learn

Implementing the math behind Linear Regression is a tedious and error-prone process. However, Scikit Learn's implementation is easy to use and flexible.

1. Create a new python file and import the following modules:
	```
	import csv
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn import linear_model
	```

2. Included in the code directory is a CSV file named [cricket.csv](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/cricket.csv). The data represents the recorded number of chirps per second for a given temperature. Let's load this data into two lists:
	```
	with open('cricket.csv', 'rb') as f:
    data = list(csv.reader(f))

	X = [[float(point[0])] for point in data]
	Y = [[float(point[1])] for point in data]
	```

3. Let's take a look at the data:
	```
	plt.scatter(X, Y, color='black')
	plt.xlabel('Temperature')
	plt.ylabel('Chirps/Sec')
	plt.show()
	```
	- If we run the script, we see the positive correlation between the temperature and chirps/sec data:
![Lin Reg Points](/images/linear_regression_points.png?raw=true "Lin Reg Points")

4. Now, using Scikit Learn's `LinearRegression` class, fit a line to the data:
	```
	regr = linear_model.LinearRegression()
	regr.fit(X, Y)
	```

5. Let's look at the line Scikit Learn produced:
	```
	plt.scatter(X, Y,  color='black')
	plt.plot(X, regr.predict(X), color='blue', linewidth=3)
	plt.xlabel('Temperature')
	plt.ylabel('Chirps/Sec')
	plt.show()
	```
	- Running this script will produce the following figure:
![Lin Reg Fit](/images/linear_regression_fit.png?raw=true "Lin Reg Fit")
	- The blue line in this figure represents the line that the `LinearRegression` class fit to the data. We can see how this line is a good representation of the data. Now, we can use this line to predict the number of chirps per second for a given temperature within the bounds of our temperature data.

The full code for this example can be found [here](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/regression.py). The data can be found [here](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/cricket.csv).

Up Next: [Clustering](https://github.com/rpcrimi/Scikit_Learn/blob/master/markdown/clustering.md)
