import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

with open('cricket.csv', 'rb') as f:
    data = list(csv.reader(f))

X = [[float(point[0])] for point in data]
Y = [[float(point[1])] for point in data]

plt.scatter(X, Y, color='black')
plt.xlabel('Temperature')
plt.ylabel('Chirps/Sec')
plt.show()

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("Residual sum of squares: %.2f" % np.mean((regr.predict(X) - Y) ** 2))
print('Variance score: %.2f' % regr.score(X, Y))

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.plot(X, regr.predict(X), color='blue', linewidth=3)
plt.xlabel('Temperature')
plt.ylabel('Chirps/Sec')
plt.show()