# Machine Learning

> "Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed."
> Arthur Samuel

Machine Learning (ML) is a subfield of computer science that facilitates the prediction of future events and decision-making. Similar to Artificial Intelligence, it attempts to build real-world knowledge to improve computers performance. Unlike AI, ML focuses more on the statistical representations of data to develop self learning algorithms.

## Fields of Machine Learning
![Comic](/images/supervised_unsupervised.png?raw=true "Comic")

Machine Learning encompasses three types of self learning algorithms:

1. Supervised Learning (SL)
	- SL is the most common application of ML, as we, the users, often just want an algorithm to simple classify a data point. SL algorithms have prior knowledge of the structure of the problem at hand. They are tasked with learning from this knowledge to better map new inputs to outputs. For example, Spam Detection is an SL technique. We can train a Supervised Learning algorithm with emails already classified as Span or Non-Spam, and then ask it to classify new emails based on what it has learned from the training.
	- Examples:
		- K-Nearest Neighbors
		- Boosting
		- Naive Bayes
		- Logistic Regression
		- Support Vector Machines
2. Unsupervised Learning (UL)
	- UL algorithms have no prior knowledge of the structure of the problem at hand. UL algorithms must learn the underlying structure and hidden patterns of the data. For example, to cluster data into groups, an UL algorithm will have no prior knowledge of the structure of these groups, but will attempt to learn the structure by analyzing the data.
	- Examples:
		- Clustering
		- Neural Networks
		- Topic Models
		- Gibbs Sampling
		- Association Rule Learning
3. Reinforcement Learning
	- An RL algorithm interacts with a dynamic environment and has no knowledge of if it correctly performed a given task. For example, self-driving cars use RL as they reinforce their knowledge as they are driving.

In this presentation, we will focus on Supervised and Unsupervised Learning.

Up Next: [Set Up](https://github.com/rpcrimi/Scikit_Learn/blob/master/markdown/set_up.md)