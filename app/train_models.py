from train_model_utils import *
from sklearn.datasets import fetch_covtype
from keras.models import load_model

# load the data
data = fetch_covtype()
# split the data
X_train, X_test, y_train, y_test = split_data(data)

# save list of feature names for later use to pickle
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(data.feature_names, f)

# convert the data to pandas dataframe for heuristic model
x_train = pd.DataFrame(X_train, columns=data.feature_names)
x_test = pd.DataFrame(X_test, columns=data.feature_names)

# train heuristic model
hm = heuristic_model(x_train, y_train)

# train decision tree
dt = decision_tree(X_train, y_train)

# train random forest
rm = random_forest(X_train, y_train)

# train neural network
# nn, nn_history = neural_network(X_train, y_train)
# load the best model
nn = load_model('models/neural_network.h5')

# Plot training curves for the best hyperparameter
# plot_training_curves(nn.history)
# evaluate the models
hm_score = evaluate_model(hm, x_test, y_test, "Heuristic-Model")
dt_score = evaluate_model(dt, X_test, y_test, "Decision-Tree")
rm_score = evaluate_model(rm, X_test, y_test, "Random-Forest")
nn_score = evaluate_model(nn, X_test, y_test, "Neural-Network")

models = ["Heuristic-Model", dt, rm, nn]
compare_models(models, X_test, y_test, x_test)
