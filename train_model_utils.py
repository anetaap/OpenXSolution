import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# define a function that will split the data
def split_data(data, test_size=0.2, random_state=42):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# define a function that will train a decision tree model
def decision_tree(X_train, y_train, max_depth=None, random_state=None):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    # save the model to pickle file
    pickle.dump(dt, open('models/decision_tree.pkl', 'wb'))
    return dt


# define a function that will train a random forest model
def random_forest(X_train, y_train, n_estimators=None, random_state=None):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    # save the model
    pickle.dump(rf, open('models/random_forest.pkl', 'wb'))
    return rf


# define a function that will train a neural network model
def neural_network(X_train, y_train, layers=2, units=64, activation='relu', optimizer='adam', loss='binary_crossentropy',
                       metrics=['accuracy'], epochs=10, nr=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=units, activation=activation, input_dim=X_train.shape[1]))
    for i in range(layers - 1):
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(X_train, y_train, epochs=epochs)
        # save the model
        model.save(f'models/neural_network{nr}.h5')
        return model


# define a function that will make predictions
def predict(model, X_test):
    return model.predict(X_test)
