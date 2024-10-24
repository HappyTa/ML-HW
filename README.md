# READ ME

## How to use this repository

### The What

This repository is made to hold all my answers for an assignment from my Machine Learning class, the questions is in the file: ```princML_hw02-1.pd```

| File Name | What they are |
|-----------|---------------|
| [answers.md](./answers.md)| Contain all non-code answers for the assignment|
| [main.py](./main.py) | This contain code related answers for question 1d and 2|
| [ml.py](./ml.py) | Contain the implementation of Linear and logistic regressions |
| [test.py](./test.py) | This contain all the code used to generate the graph for question 3A |
| [feature_engineering.py](./feature_engineering.py) | This contain all the code used tot generate the graphs for question 3B |

### How?

This section will talk about what you need to run the scripts and how to run them.

#### Dependencies

To run this you will need:

1. [scikit learn](https://pypi.org/project/scikit-learn/)

```
pip install scikit-learn
```

2. [numpy](https://pypi.org/project/numpy/)

```
pip install numpy
```

3. [Matplotlib](https://pypi.org/project/matplotlib/)

```
pip install matplotlib
```

4. [SciPy](https://pypi.org/project/scipy/)

```
pip install scipy
```

#### Runnning main.py

This function allow the user to select what they want to run by using messages print onto the commandline. It require 1 arguemnt to be passed in which represent type of algorithm they would like to run:

```
python3 -O main.py [argument]

-O         Turn off debug mode (this will supress debug messages)
[arguemnt] Must be of int type
           0: Linear Regression
           1: Logistic Regression
```

The main purpose of this script is to demonstrate how the algorithm can be use.

#### Running test.py

This script is made to test the Linear Regression prediction function

```
python3 -O test.py
-O         Turn off debug mode (this will supress debug messages)
```

This will return a graph, detailing the Root Mean Square Error result from the algorithm depending on the percentage of data used for training.

#### Runnning feature_engineering.py

This script is made to test the accuracy of my Logistic Regression implementation. It require an argument to be passed in.

```
python3 -O feature_engineering [arguemnt]

-O         Turn off debug mode (this will supress debug messages)
[arguemnt] Must be of int type
           0: Graph for question 3A
           1: Graph for questiong 3B
```

 This return 2 differnt sets of graphs based on what arguemnt was passed in. If 0 was passed in, it will spit out the confusion matrix for each subsets of the training data. If 1 was passed in it will return the accuracy, precision, recall rate, and F1-score for each of the data subset.H

#### Other notes

I have marked where you can manually set the hyper-parameters using **TODO** markers. By default, only main.py allow for user input to determine parameters, this feature was not implemented for the other sections to save time.
