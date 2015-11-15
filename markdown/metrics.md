# Metrics

It is important to have metrics about your data and algorithm throughout your development process so that you can gauge how your algorithm is performing. One important advantage of using Scikit Learn is its metrics features, which are much more powerful than those provided by other Python ML packages. In this section, we will cover two important Scikit Learn's metrics features: Confusion Matrices and Classification Scores.

## Metrics in Scikit Learn
In this section, we will work with the digits dataset provided by Scikit Learn. This dataset includes thousands of images of handwritten digits including each ones actual value. From this dataset, we can use an SVM to classify test data. The SVM will produce a confusion matrix and classification scores.

1. Create a new python file and import the following modules:
	```
	from sklearn import datasets, svm, metrics
	```

2. Load the digits dataset:
	```
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	```

3. Use Scikit Learn's `SVM` class to fit the training data:
	```
	classifier = svm.SVC(gamma=0.001)
	classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
	```

4. Split the training data in half so that we can use the second half as testing data:
	```
	expected = digits.target[n_samples / 2:]
	predicted = classifier.predict(data[n_samples / 2:])
	```

5. Now we can use Scikit Learn's metrics (`classification_report` and `confusion_matrix`) to see how our algorithm is performing:
	```
	print("Classification Report:\n%s\n" % (metrics.classification_report(expected, predicted)))
	print("Confusion Matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
	```

6. Run the script. You should see the following output:
	```
	Classification Report:
             precision    recall  f1-score   support

          0       1.00      0.99      0.99        88
          1       0.99      0.97      0.98        91
          2       0.99      0.99      0.99        86
          3       0.98      0.87      0.92        91
          4       0.99      0.96      0.97        92
          5       0.95      0.97      0.96        91
          6       0.99      0.99      0.99        91
          7       0.96      0.99      0.97        89
          8       0.94      1.00      0.97        88
          9       0.93      0.98      0.95        92

avg / total       0.97      0.97      0.97       899


Confusion Matrix:
[[87  0  0  0  1  0  0  0  0  0]
 [ 0 88  1  0  0  0  0  0  1  1]
 [ 0  0 85  1  0  0  0  0  0  0]
 [ 0  0  0 79  0  3  0  4  5  0]
 [ 0  0  0  0 88  0  0  0  0  4]
 [ 0  0  0  0  0 88  1  0  0  2]
 [ 0  1  0  0  0  0 90  0  0  0]
 [ 0  0  0  0  0  1  0 88  0  0]
 [ 0  0  0  0  0  0  0  0 88  0]
 [ 0  0  0  1  0  1  0  0  0 90]]
 	```
 	- The Classification Report gives information about how well each number in the test set was classified. This includes information about classification precision and recall which lets developers know the accuracy of their classifier.
 	- The Confusion Matrix is a representation of which numbers classified as other numbers. A good classifier will have large values along the diagonal of the matrix, as we see above. In this matrix, our SVM classified a test point representing the number 10 correctly 90 times. On the other hand, it classified a test point representing the number 3 as a 9 five times.

The full code for this example can be found [here](https://github.com/rpcrimi/Scikit_Learn/blob/master/code/metrics.py)

Up Next: [Conclusion](https://github.com/rpcrimi/Scikit_Learn/blob/master/markdown/conclusion.md)
