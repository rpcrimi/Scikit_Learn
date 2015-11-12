# Linear Regression

Linear Regression is a technique in which semi-linear data is fit to a linear line. Once we fit this line, we can predict results corresponding to different input values. We can see an example of Linear Regression in the figure below:

![Linear Regression](/images/linear_regression_example.jpg?raw=true "Linear Regression")

To calculate this line, we want to minimize the residual sum of squares. This means that the line will be as close as it can be to each point in the dataset.

## SVMs in Scikit Learn

While the math behind Linear Regression is fairly simple, Scikit Learn's implementation is easy to use and flexible.

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
	- If we run the script, we see the positive correlation between temperature and chirps/sec:
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
	- The blue line in this figure represents the line that the `LinearRegression` class fit to the data. We can see how this line is a good representation of the data. Now, we can use this line to predict the number of chirps per second for a temperature that we have not observed.

The full code for this example can be found [here](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/regression.py). The data can be found [here](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/cricket.csv).