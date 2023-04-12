from train_model_utils import *
from sklearn.datasets import fetch_covtype

# load the data
data = fetch_covtype()
# split the data
X_train, X_test, y_train, y_test = split_data(data)
# hm = ml.heuristic_model()

# train decision tree
dt = decision_tree(X_train, y_train)

# train random forest
rm = random_forest(X_train, y_train)

# train first neural network
nn1 = neural_network(X_train, y_train, nr=1)

# train second neural network
nn2 = neural_network(X_train, y_train, nr=2)
