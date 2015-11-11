# Machine Learning

> "Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed."
> Arthur Samuel

Machine Learning is a subfield of computer science in which algorithms build models from data to help in the predictions and decisions of future events. Compared to Artificial Intelligence, Machine Learning focuses more on the statistical representations of data to develop self learning algorithms.

## Fields of Machine Learning
![Comic](/images/supervised_unsupervised.png?raw=true "Comic")

There are three types of self learning algorithms within Machine Learning:

1. Supervised Learning
	- Such algorithms have prior knowledge of the structure of the problem and are tasked with learning from this knowledge to map new inputs to outputs. For example, Spam Detection is a supervised learning technique. We can provide a Supervised Learning algorithm with examples of Spam and Non-Spam emails and then ask it to classify new emails as one of the two. Supervised Learning is the most common application of Machine Learning as we usually want to know "what" a data point is.
	- Examples:
		- K-Nearest Neighbors
		- Boosting
		- Naive Bayes
		- Logistic Regression
		- Support Vector Machines
2. Unsupervised Learning
	- Such algorithms have no prior knowledge of the structure of the problem and are tasked with learning the underlying structure and hidden patterns of the data. For example, clustering data into groups is an example of Unsupervised Learning. The algorithm will have no prior knowledge of the structure of these groups, but will attempt to learn the structure by analyzing the data.
	- Examples:
		- Clustering
		- Neural Networks
		- Topic Models
		- Gibbs Sampling
		- Association Rule Learning
3. Reinforcement Learning
	- Such algorithms interact with a dynamic environment and have no knowledge of whether they correctly performed a given task. For example, self-driving cars use Reinforcement Learning as they reinforce their knowledge as they are driving.

In this presentation, we will only focus on Supervised and Unsupervised Learning.