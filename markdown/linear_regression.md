# Linear Regression

Linear Regression is a technique in which semi-linear data is fit to a linear line. We can see an example of Linear Regression in the figure below:

![Linear Regression](/images/linear_regression_example.jpg?raw=true "Linear Regression")

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

2. Included in the code directory is a CSV file named [cricket.csv](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/cricket.csv). The data represents the recorded number of chirps per second for a given temperature. Let's load this data into two lists:
	```
	with open('cricket.csv', 'rb') as f:
    data = list(csv.reader(f))

	X = [[float(point[0])] for point in data]
	Y = [[float(point[1])] for point in data]
	```

3. If we plot this data, :
	```
	plt.scatter(X, Y, color='black')
	plt.xlabel('Temperature')
	plt.ylabel('Chirps/Sec')
	plt.show()
	```
	- If we run the script, we see the positive correlation between temperature and chirps/sec:
![Lin Reg Points](/images/linear_regression_points.png?raw=true "Lin Reg Points")