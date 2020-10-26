# GroupProj_Engine
How to Connect Model Input Data With Predictions for Machine Learning
by Jason Brownlee on November 15, 2019 in Python Machine Learning
Tweet  Share
Last Updated on August 19, 2020

Fitting a model to a training dataset is so easy today with libraries like scikit-learn.

A model can be fit and evaluated on a dataset in just a few lines of code. It is so easy that it has become a problem.

The same few lines of code are repeated again and again and it may not be obvious how to actually use the model to make a prediction. Or, if a prediction is made, how to relate the predicted values to the actual input values.

I know that this is the case because I get many emails with the question:

How do I connect the predicted values with the input data?

This a common problem.

In this tutorial, you will discover how to relate the predicted values with the inputs to a machine learning model.

After completing this tutorial, you will know:

How to fit and evaluate the model on a training dataset.
How to use the fit model to make predictions one at a time and in batches.
How to connect the predicted values with the inputs to the model.
Kick-start your project with my new book Machine Learning Mastery With Python, including step-by-step tutorials and the Python source code files for all examples.

Let’s get started.

Update Jan/2020: Updated for changes in scikit-learn v0.22 API.
How to Connect Model Input Data With Predictions for Machine Learning
How to Connect Model Input Data With Predictions for Machine Learning
Photo by Ian D. Keating, some rights reserved.

Tutorial Overview
This tutorial is divided into three parts; they are:

Prepare a Training Dataset
How to Fit a Model on the Training Dataset
How to Connect Predictions With Inputs to the Model
Prepare a Training Dataset
Let’s start off by defining a dataset that we can use with our model.

You may have your own dataset in a CSV file or in a NumPy array in memory.

In this case, we will use a simple two-class or binary classification problem with two numerical input variables.

Inputs: Two numerical input variables:
Outputs: A class label as either a 0 or 1.
We can use the make_blobs() scikit-learn function to create this dataset with 1,000 examples.

The example below creates the dataset with separate arrays for the input (X) and outputs (y).

# example of creating a test dataset
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# summarize the shape of the arrays
print(X.shape, y.shape)
1
2
3
4
5
6
# example of creating a test dataset
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# summarize the shape of the arrays
print(X.shape, y.shape)
Running the example creates the dataset and prints the shape of each of the arrays.

We can see that there are 1,000 rows for the 1,000 samples in the dataset. We can also see that the input data has two columns for the two input variables and that the output array is one long array of class labels for each of the rows in the input data.

(1000, 2) (1000,)
1
(1000, 2) (1000,)
Next, we will fit a model on this training dataset.

How to Fit a Model on the Training Dataset
Now that we have a training dataset, we can fit a model on the data.

This means that we will provide all of the training data to a learning algorithm and let the learning algorithm to discover the mapping between the inputs and the output class label that minimizes the prediction error.

In this case, because it is a two-class problem, we will try the logistic regression classification algorithm.

This can be achieved using the LogisticRegression class from scikit-learn.

First, the model must be defined with any specific configuration we require. In this case, we will use the efficient ‘lbfgs‘ solver.

Next, the model is fit on the training dataset by calling the fit() function and passing in the training dataset.

Finally, we can evaluate the model by first using it to make predictions on the training dataset by calling predict() and then comparing the predictions to the expected class labels and calculating the accuracy.

The complete example is listed below.

# fit a logistic regression on the training dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
# evaluate predictions
acc = accuracy_score(y, yhat)
print(acc)
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
# fit a logistic regression on the training dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
# evaluate predictions
acc = accuracy_score(y, yhat)
print(acc)
Running the example fits the model on the training dataset and then prints the classification accuracy.

In this case, we can see that the model has a 100% classification accuracy on the training dataset.

1.0
1
1.0
Now that we know how to fit and evaluate a model on the training dataset, let’s get to the root of the question.

How do you connect inputs of the model to the outputs?

How to Connect Predictions With Inputs to the Model
A fit machine learning model takes inputs and makes a prediction.

This could be one row of data at a time; for example:

Input: 2.12309797 -1.41131072
Output: 1
This is straightforward with our model.

For example, we can make a prediction with an array input and get one output and we know that the two are directly connected.

The input must be defined as an array of numbers, specifically 1 row with 2 columns. We can achieve this by defining the example as a list of rows with a list of columns for each row; for example:

...
# define input
new_input = [[2.12309797, -1.41131072]]
1
2
3
...
# define input
new_input = [[2.12309797, -1.41131072]]
We can then provide this as input to the model and make a prediction.

...
# get prediction for new input
new_output = model.predict(new_input)
1
2
3
...
# get prediction for new input
new_output = model.predict(new_input)
Tying this together with fitting the model from the previous section, the complete example is listed below.

# make a single prediction with the model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# define input
new_input = [[2.12309797, -1.41131072]]
# get prediction for new input
new_output = model.predict(new_input)
# summarize input and output
print(new_input, new_output)
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
# make a single prediction with the model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# define input
new_input = [[2.12309797, -1.41131072]]
# get prediction for new input
new_output = model.predict(new_input)
# summarize input and output
print(new_input, new_output)
Running the example defines the new input and makes a prediction, then prints both the input and the output.

We can see that in this case, the model predicts class label 1 for the inputs.

[[2.12309797, -1.41131072]] [1]
1
[[2.12309797, -1.41131072]] [1]
If we were using the model in our own application, this usage of the model would allow us to directly relate the inputs and outputs for each prediction made.

If we needed to replace the labels 0 and 1 with something meaningful like “spam” and “not spam“, we could do that with a simple if-statement.

So far so good.

What happens when the model is used to make multiple predictions at once?

That is, how do we relate the predictions to the inputs when multiple rows or multiple samples are provided to the model at once?

For example, we could make a prediction for each of the 1,000 examples in the training dataset as we did in the previous section when evaluating the model. In this case, the model would make 1,000 distinct predictions and return an array of 1,000 integer values. One prediction for each of the 1,000 input rows of data.

Importantly, the order of the predictions in the output array matches the order of rows provided as input to the model when making a prediction. This means that the input row at index 0 matches the prediction at index 0; the same is true for index 1, index 2, all the way to index 999.

Therefore, we can relate the inputs and outputs directly based on their index, with the knowledge that the order is preserved when making a prediction on many rows of inputs.

Let’s make this concrete with an example.

First, we can make a prediction for each row of input in the training dataset:

...
# make predictions on the entire training dataset
yhat = model.predict(X)
1
2
3
...
# make predictions on the entire training dataset
yhat = model.predict(X)
We can then step through the indexes and access the input and the predicted output for each.

This shows precisely how to connect the predictions with the input rows. For example, the input at row 0 and the prediction at index 0:

...
print(X[0], yhat[0])
1
2
...
print(X[0], yhat[0])
In this case, we will just look at the first 10 rows and their predictions.

...
# connect predictions with outputs
for i in range(10):
	print(X[i], yhat[i])
1
2
3
4
...
# connect predictions with outputs
for i in range(10):
	print(X[i], yhat[i])
Tying this together, the complete example of making a prediction for each row in the training data and connecting the predictions with the inputs is listed below.

# make a single prediction with the model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# make predictions on the entire training dataset
yhat = model.predict(X)
# connect predictions with outputs
for i in range(10):
	print(X[i], yhat[i])
1
2
3
4
5
6
7
8
9
10
11
12
13
14
# make a single prediction with the model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# make predictions on the entire training dataset
yhat = model.predict(X)
# connect predictions with outputs
for i in range(10):
	print(X[i], yhat[i])
Running the example, the model makes 1,000 predictions for the 1,000 rows in the training dataset, then connects the inputs to the predicted values for the first 10 examples.

This provides a template that you can use and adapt for your own predictive modeling projects to connect predictions to the input rows via their row index.

[ 1.23839154 -2.8475005 ] 1
[-1.25884111 -8.57055785] 0
[ -0.86599821 -10.50446358] 0
[ 0.59831673 -1.06451727] 1
[ 2.12309797 -1.41131072] 1
[-1.53722693 -9.61845366] 0
[ 0.92194131 -0.68709327] 1
[-1.31478732 -8.78528161] 0
[ 1.57989896 -1.462412  ] 1
[ 1.36989667 -1.3964704 ] 1
1
2
3
4
5
6
7
8
9
10
[ 1.23839154 -2.8475005 ] 1
[-1.25884111 -8.57055785] 0
[ -0.86599821 -10.50446358] 0
[ 0.59831673 -1.06451727] 1
[ 2.12309797 -1.41131072] 1
[-1.53722693 -9.61845366] 0
[ 0.92194131 -0.68709327] 1
[-1.31478732 -8.78528161] 0
[ 1.57989896 -1.462412  ] 1
[ 1.36989667 -1.3964704 ] 1
Further Reading
This section provides more resources on the topic if you are looking to go deeper.

Posts
Your First Machine Learning Project in Python Step-By-Step
How to Make Predictions with scikit-learn
APIs
sklearn.datasets.make_blobs API
sklearn.metrics.accuracy_score API
sklearn.linear_model.LogisticRegression API
Summary
In this tutorial, you discovered how to relate the predicted values with the inputs to a machine learning model.

Specifically, you learned:

How to fit and evaluate the model on a training dataset.
How to use the fit model to make predictions one at a time and in batches.
How to connect the predicted values with the inputs to the model.
Do you have any questions?
Ask your questions in the comments below and I will do my best to answer.

