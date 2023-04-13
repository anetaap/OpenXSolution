import pickle

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint


# define a function that will split the data
def split_data(data, test_size=0.2, random_state=42):
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# define rules for classifying samples
def classify(x):
    if x['Elevation'] <= 2900:
        return 1
    elif x['Aspect'] <= 150:
        return 2
    elif x['Slope'] <= 10:
        return 3
    elif x['Horizontal_Distance_To_Roadways'] <= 400:
        return 4
    elif x['Hillshade_9am'] <= 215:
        return 5
    elif x['Hillshade_Noon'] <= 225:
        return 6
    else:
        return 7


# define a heuristic model that will classify all samples form dataset
def heuristic_model(X_train, y_train):
    # classify all samples
    predictions = X_train.apply(classify, axis=1)

    # print the accuracy
    print(f'Heuristic model accuracy: {accuracy_score(y_train, predictions)}')

    # save the classify function to pickle file
    pickle.dump(classify, open('models/heuristic_model.pkl', 'wb'))
    return predictions


# define a function that will train a decision tree model
def decision_tree(X_train, y_train, max_depth=None, random_state=None):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)

    # print the accuracy
    print(f'Decision tree accuracy: {dt.score(X_train, y_train)}')

    # save the model to pickle file
    pickle.dump(dt, open('models/decision_tree.pkl', 'wb'))
    return dt


# define a function that will train a random forest model
def random_forest(X_train, y_train, n_estimators=5, random_state=2):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    # print the accuracy
    print(f'Random forest accuracy: {rf.score(X_train, y_train)}')

    # save the model
    pickle.dump(rf, open('models/random_forest.pkl', 'wb'))
    return rf


# define the Keras model
def create_model(optimizer='adam', hidden_layers=1, units=54, X_train=None):
    model = Sequential()
    for i in range(hidden_layers):
        if i == 0:
            model.add(Dense(units=units, activation='relu', input_dim=X_train.shape[1]))
        else:
            model.add(Dense(units=units, activation='relu'))
    model.add(Dense(units=7, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


#  function that will find a good set of hyperparameters for the NN model
def find_best_hyperparameters(X_train, y_train):
    # create a KerasClassifier for use with GridSearchCV
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # define the parameter grid
    param_grid = {
        'optimizer': ['adam', 'rmsprop'],
        'hidden_layers': [3, 4, 5],
        'units': [16, 32, 64],
        'batch_size': [64],
        'epochs': [40],
    }

    # define a checkpoint to save the best model
    checkpoint = ModelCheckpoint('models/neural_network.h5', monitor='accuracy', mode='max', save_best_only=True, verbose=1)

    # perform grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)
    grid_result = grid.fit(X_train, y_train, callbacks=[checkpoint])

    # print the best results
    print("Best: {:.2f}% using {}".format(grid_result.best_score_*100, grid_result.best_params_))

    return grid_result.best_estimator_.model


# define a function that will train a neural network model
def neural_network(X_train, y_train):
    model = find_best_hyperparameters(X_train, y_train)
    history = model.fit(X_train, y_train)

    # print the accuracy
    print(f'Neural network accuracy: {model.score(X_train, y_train)}')

    # save the model
    pickle.dump(model, open('models/neural_network.pkl', 'wb'))
    return model, history


# define a function that will make predictions
def predict(model, X_test):
    return model.predict(X_test)


# plot training curves for the neural network model and save them to a file
def plot_training_curves(history):
    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    plt.savefig('images/training_curves.png')
    plt.show()


# define a function that will evaluate the model
def evaluate_model(model, X_test, y_test, model_name):
    # make predictions
    if model_name == 'Heuristic-Model':
        predictions = X_test.apply(classify, axis=1)
    else:
        predictions = predict(model, X_test)
    # calculate the accuracy
    if model_name == 'Neural-Network':
        accuracy = model.evaluate(X_test, y_test)[1]
        predictions = np.argmax(predictions, axis=1)
    else:
        accuracy = accuracy_score(y_test, predictions)
    # calculate the precision
    precision = precision_score(y_test, predictions, average='weighted')
    # calculate the recall
    recall = recall_score(y_test, predictions, average='weighted')
    # calculate the f1 score
    f1 = f1_score(y_test, predictions, average='weighted')
    # save the results
    results = pd.DataFrame({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, index=[0])
    results.to_csv(f'results/{model_name}.csv', index=False)
    return results


# define a function that will compare the models
def compare_models(models, X_test, y_test, x_test):
    # create a dataframe to store the results
    results = pd.DataFrame(columns=['model', 'accuracy'])
    # loop over the models
    for model in models:
        # make predictions
        if model == 'Heuristic-Model':
            predictions = x_test.apply(classify, axis=1)
        else:
            predictions = predict(model, X_test)
        # calculate the accuracy
        # check if the model is a neural network model
        if isinstance(model, Model):
            accuracy = model.evaluate(X_test, y_test)[1]
            predictions = np.argmax(predictions, axis=1)
        else:
            accuracy = accuracy_score(y_test, predictions)
        # calculate the precision
        precision = precision_score(y_test, predictions, average='weighted')
        # calculate the recall
        recall = recall_score(y_test, predictions, average='weighted')
        # calculate the f1 score
        f1 = f1_score(y_test, predictions, average='weighted')
        # save the results
        results = results.append({'model': model.__class__.__name__, 'accuracy': accuracy, 'precision': precision,
                                    'recall': recall, 'f1': f1}, ignore_index=True)
        # save the results
        results.to_csv('results/models_comparison.csv', index=False)
    return results
