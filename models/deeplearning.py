from pandas import read_csv
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import backend as K
import numpy as np


# from utils import encode_numeric_score, to_xy
TRAIN_DATA = "./features/extracted_data/train_preprocessing_output.csv"

data = pd.read_csv(TRAIN_DATA)
Y = data['resale_price'].values
X = data.drop('resale_price', axis=1).values


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(101, input_dim=98, activation="relu"))
    model.add(Dense(60, activation="relu"))
    model.add(Dense(1))
    model.summary()  # Print model Summary
    # compile
    model.compile(loss=root_mean_squared_error, optimizer="adam",
                  metrics=[root_mean_squared_error, "mean_squared_error"])
    return model


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(
    build_fn=wider_model, epochs=100, batch_size=10, verbose=1)))
pipeline = Pipeline(estimators)

TEST_DATA = "./features/extracted_data/test_preprocessing_output.csv"
PREDICTION_OUTPUT = "./results_analysis/results/experiment_output.csv"

test_data = pd.read_csv(TEST_DATA)

test_data['block'] = test_data['block'].str.replace(
    '[A-Za-z]', '', regex=True)
test_data['block'] = test_data['block'].apply(int)

# training the model using best parameters
pipeline.fit(X, Y)


# actual prediction using model
test_predictions = pipeline.predict(test_data)

# saving data to csv
result = pd.DataFrame(
    {'Id': np.arange(len(test_predictions)), 'Predicted': test_predictions})
result.to_csv(PREDICTION_OUTPUT, index=False)
