# Stock_predictions_clustering_algorithms
## INTRODUCTION

Stock prediction aims to predict the future trends of a stock in order to help investors make good investment decisions. Traditional solutions for stock prediction are based on time-series models. With the recent success of deep neural networks in modeling sequential data, deep learning has become a promising choice for stock prediction. However, most existing deep learning solutions are not optimized toward the target of investment, i.e., selecting the best stock with the highest expected revenue. Specifically, they typically formulate stock prediction as a classification (to predict stock trends) or a regression problem (to predict stock prices). More importantly, they largely treat the stocks as independent of each other. The valuable signal in the rich relations between stocks (or companies), such as two stocks are in the same sector and two companies have a supplier-customer relation, is not considered.

![](RackMultipart20220427-1-1a98p4_html_139c8d7c90f15b5.png)

The important terminologies to know before digging into this project is as follows:

1. Price - Current Price
2. Open - Opening Price Of Stock For The Day
3. Close - Closing Price Of Stock For The Day
4. High - Highest Price Of Stock For The Day
5. Low - Lowest Price Of Stock For The Day

The procedure of approaching stock prediction as a classification problem is as follows:

1. Get the input for the algorithm includes (IV, independent variables):
  1. Moving average of historical close prices
  2. Trading volume, and open, highest, lowest, and close prices of the present trading
2. Perform Data cleaning and Normalizing the input
3. Train the classifier
4. Predict based on the classification of the model:
  1. the stock price will rise (1)
  2. the stock price will fall (-1)

## ALGORITHMS

## Naive Bayes

Bayes&#39; Theorem provides a way that we can calculate the probability of a piece of data belonging to a given class, given our prior knowledge. Bayes&#39; Theorem is stated as:

P(class|data) = (P(data|class) \* P(class)) / P(data)

Where P(class|data) is the probability of class given the provided data.

Naive Bayes is a classification algorithm for binary (two-class) and multiclass classification problems. It is called Naive Bayes or idiot Bayes because the calculations of the probabilities for each class are simplified to make their calculations tractable. Rather than attempting to calculate the probabilities of each attribute value, they are assumed to be conditionally independent given the class value. This is a very strong assumption that is most unlikely in real data, i.e. that the attributes do not interact. Nevertheless, the approach performs surprisingly well on data where this assumption does not hold.

This Naive Bayes algorithm is broken down into 5 parts:

- Step 1: Separate By Class.
- Step 2: Summarize Dataset.
- Step 3: Summarize Data By Class.
- Step 4: Gaussian Probability Density Function.
- Step 5: Class Probabilities.

### Step 1: Separate By Class

We will need to calculate the probability of data by the class they belong to, the so-called base rate. This means that we will first need to separate our training data by class. A relatively straightforward operation. We can create a dictionary object where each key is the class value and then add a list of all the records as the value in the dictionary.

### Step 2: Summarize Dataset

Calculate the mean, standard deviation and count for each column in a dataset

### Step 3: Summarize Data By Class

Computing the std, mean and count for every class value, here up and down

### Step 4: Gaussian Probability Density Function

A Gaussian distribution can be summarized using only two numbers: the mean and the standard deviation. Therefore, with a little math, we can estimate the probability of a given value. This piece of math is called a Gaussian Probability Distribution Function (or Gaussian PDF) and can be calculated as:

f(x) = (1 / sqrt(2 \* PI) \* sigma) \* exp(-((x-mean)^2 / (2 \* sigma^2)))

### Step 5: Class Probabilities

Now it is time to use the statistics calculated from our training data to calculate probabilities for new data. Probabilities are calculated separately for each class. This means that we first calculate the probability that a new piece of data belongs to the first class, then calculate probabilities that it belongs to the second class, and so on for all the classes. The probability that a piece of data belongs to a class is calculated as follows:

**P(class|data) = P(X|class) \* P(class)**

You may note that this is different from the Bayes Theorem described above. The division has been removed to simplify the calculation. This means that the result is no longer strictly a probability of the data belonging to a class. The value is still maximized, meaning that the calculation for the class that results in the largest value is taken as the prediction. This is a common implementation simplification as we are often more interested in the class prediction rather than the probability.

## KNN

The k-Nearest Neighbors algorithm or KNN for short is a very simple technique. The entire training dataset is stored. When a prediction is required, the k-most similar records to a new record from the training dataset are then located. From these neighbors, a summarized prediction is made. Similarity between records can be measured many different ways. A problem or data-specific method can be used. Generally, with tabular data, a good starting point is the Euclidean distance. Once the neighbors are discovered, the summary prediction can be made by returning the most common outcome or taking the average. As such, KNN can be used for classification or regression problems. There is no model to speak of other than holding the entire training dataset. Because no work is done until a prediction is required, KNN is often referred to as a lazy learning method.

This k-Nearest Neighbors is broken down into 3 parts:

- Step 1: Calculate Euclidean Distance.
- Step 2: Get Nearest Neighbors.
- Step 3: Make Predictions.

### Step 1: Calculate Euclidean Distance

The first step is to calculate the distance between two rows in a dataset. Rows of data are mostly made up of numbers and an easy way to calculate the distance between two rows or vectors of numbers is to draw a straight line. This makes sense in 2D or 3D and scales nicely to higher dimensions. We can calculate the straight line distance between two vectors using the Euclidean distance measure. It is calculated as the square root of the sum of the squared differences between the two vectors.

**Euclidean Distance =**

Where x1 is the first row of data, x2 is the second row of data and &quot;i&quot; is the index to a specific column as we sum across all columns. With Euclidean distance, the smaller the value, the more similar two records will be.

### Step 2: Get Nearest Neighbors

Neighbors for a new piece of data in the dataset are the k closest instances, as defined by our distance measure. To locate the neighbors for a new piece of data within a dataset we must first calculate the distance between each record in the dataset to the new piece of data. We can do this using our distance function prepared above. Once distances are calculated, we must sort all of the records in the training dataset by their distance to the new data. We can then select the top k to return as the most similar neighbors. We can do this by keeping track of the distance for each record in the dataset as a tuple, sort the list of tuples by the distance (in descending order) and then retrieve the neighbors.

### Step 3: Make Predictions

The most similar neighbors collected from the training dataset can be used to make predictions. In the case of classification, we can return the most represented class among the neighbors. We can achieve this by finding the most occurring class on the list of output values from the neighbors.

## Decision Tree [CART]

Classification and Regression Trees or CART for short is an acronym introduced by Leo Breiman to refer to Decision Tree algorithms that can be used for classification or regression predictive modeling problems. The representation of the CART model is a binary tree. This is the same binary tree from algorithms and data structures, nothing too fancy (each node can have zero, one or two child nodes). A node represents a single input variable (X) and a split point on that variable, assuming the variable is numeric. The leaf nodes (also called terminal nodes) of the tree contain an output variable (y) which is used to make a prediction. Once created, a tree can be navigated with a new row of data following each branch with the splits until a final prediction is made. Creating a binary decision tree is actually a process of dividing up the input space. A greedy approach is used to divide the space called recursive binary splitting. This is a numerical procedure where all the values are lined up and different split points are tried and tested using a cost function. The split with the best cost (lowest cost because we minimize cost) is selected. All input variables and all possible split points are evaluated and chosen in a greedy manner based on the cost function.For classification problems the Gini cost function is used which provides an indication of how pure the nodes are, where node purity refers to how mixed the training data assigned to each node is.

Splitting continues until nodes contain a minimum number of training examples or a maximum tree depth is reached.

This algorithm is broken down into 4 parts:

- Step 1: Calculate Gini Index.
- Step 2: Create Split.
- Step 3: Build a Tree.
- Step 4: Make a Prediction.

### Step 1: Calculate Gini Index.

The Gini index is the name of the cost function used to evaluate splits in the dataset. A split in the dataset involves one input attribute and one value for that attribute. It can be used to divide training patterns into two groups of rows. A Gini score gives an idea of how good a split is by how mixed the classes are in the two groups created by the split. A perfect separation results in a Gini score of 0, whereas the worst case split that results in 50/50 classes in each group results in a Gini score of 0.5 (for a 2 class problem). The Gini index for each group must be weighted by the size of the group, relative to all of the samples in the parent.

### Step 2: Create Split

A split is composed of an attribute in the dataset and a value. We can summarize this as the index of an attribute to split and the value by which to split rows on that attribute. This is just a useful shorthand for indexing into rows of data. Creating a split involves three parts, the first we have already looked at which is calculating the Gini score. The remaining two parts are:

- Splitting a Dataset.
- Evaluating All Splits.

#### Step 2.1: Splitting a Dataset

Splitting a dataset means separating a dataset into two lists of rows given the index of an attribute and a split value for that attribute. Once we have the two groups, we can then use our Gini score above to evaluate the cost of the split. Splitting a dataset involves iterating over each row, checking if the attribute value is below or above the split value and assigning it to the left or right group respectively. Note that the right group contains all rows with a value at the index above or equal to the split value.

#### Step 2.2: Evaluating All Splits

With the Gini function above and the test split function we now have everything we need to evaluate splits. Given a dataset, we must check every value on each attribute as a candidate split, evaluate the cost of the split and find the best possible split we could make. Once the best split is found, we can use it as a node in our decision tree. This is an exhaustive and greedy algorithm. We will use a dictionary to represent a node in the decision tree as we can store data by name. When selecting the best split and using it as a new node for the tree we will store the index of the chosen attribute, the value of that attribute by which to split and the two groups of data split by the chosen split point. Each group of data is its own small dataset of just those rows assigned to the left or right group by the splitting process. You can imagine how we might split each group again, recursively as we build out our decision tree. The best split is recorded and then returned after all checks are complete.

### Step 3: Build a Tree

Building a tree may be divided into 3 main parts:

- Terminal Nodes.
- Recursive Splitting.
- Building a Tree.

#### Step 3.1: Terminal Nodes

We need to decide when to stop growing a tree. We can do that using the depth and the number of rows that the node is responsible for in the training dataset.

- Maximum Tree Depth: This is the maximum number of nodes from the root node of the tree. Once a maximum depth of the tree is met, we must stop splitting and adding new nodes. Deeper trees are more complex and are more likely to overfit the training data.
- Minimum Node Records: This is the minimum number of training patterns that a given node is responsible for. Once at or below this minimum, we must stop splitting and adding new nodes. Nodes that account for too few training patterns are expected to be too specific and are likely to overfit the training data.

There is one more condition. It is possible to choose a split in which all rows belong to one group. In this case, we will be unable to continue splitting and adding child nodes as we will have no records to split on one side or another. Now we have some ideas of when to stop growing the tree. When we do stop growing at a given point, that node is called a terminal node and is used to make a final prediction.

#### Step 3.2: Recursive Splitting

We know how and when to create terminal nodes, now we can build our tree. Building a decision tree involves calling the above developed get\_split() function over and over again on the groups created for each node. New nodes added to an existing node are called child nodes. A node may have zero children (a terminal node), one child (one side makes a prediction directly) or two child nodes. We will refer to the child nodes as left and right in the dictionary representation of a given node. Once a node is created, we can create child nodes recursively on each group of data from the split by calling the same function again. Below is a function that implements this recursive procedure. It takes a node as an argument as well as the maximum depth, minimum number of patterns in a node and the current depth of a node. You can imagine how this might be first called passing in the root node and the depth of 1. This function is best explained in steps:

1. Firstly, the two groups of data split by the node are extracted for use and deleted from the node. As we work on these groups the node no longer requires access to these data.
2. Next, we check if either the left or right group of rows is empty and if so we create a terminal node using what records we do have.
3. We then check if we have reached our maximum depth and if so we create a terminal node.
4. We then process the left child, creating a terminal node if the group of rows is too small, otherwise creating and adding the left node in a depth-first fashion until the bottom of the tree is reached on this branch.
5. The right side is then processed in the same manner, as we rise back up the constructed tree to the root.

#### Step 3.3: Building a Tree

We can now put all of the pieces together. Building the tree involves creating the root node and calling the split() function that then calls itself recursively to build out the whole tree.

### 4. Make a Prediction

Making predictions with a decision tree involves navigating the tree with the specifically provided row of data. Again, we can implement this using a recursive function, where the same prediction routine is called again with the left or the right child nodes, depending on how the split affects the provided data. We must check if a child node is either a terminal value to be returned as the prediction, or if it is a dictionary node containing another level of the tree to be considered.

# Random Forest Algorithm

Decision trees involve the greedy selection of the best split point from the dataset at each step. This algorithm makes decision trees susceptible to high variance if they are not pruned. This high variance can be harnessed and reduced by creating multiple trees with different samples of the training dataset (different views of the problem) and combining their predictions. This approach is called bootstrap aggregation or bagging for short. A limitation of bagging is that the same greedy algorithm is used to create each tree, meaning that it is likely that the same or very similar split points will be chosen in each tree making the different trees very similar (trees will be correlated). This, in turn, makes their predictions similar, mitigating the variance originally sought. We can force the decision trees to be different by limiting the features (rows) that the greedy algorithm can evaluate at each split point when creating the tree. This is called the Random Forest algorithm. Like bagging, multiple samples of the training dataset are taken and a different tree trained on each. The difference is that at each point a split is made in the data and added to the tree, only a fixed subset of attributes can be considered.


# REFERENCES
[For KNN](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
[For Naive bayes](https://machinelearningmastery.com/naive-bayes-for-machine-learning/)
[For Decision tree](https://machinelearningmastery.com/random-forest-ensemble-in-python/)
[For Random forest](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)
