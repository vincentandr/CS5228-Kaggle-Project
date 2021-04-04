from pandas import read_csv
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
# dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# dataset = dataframe.values
# split into input (X) and output (Y) variables
# X = dataset[:, 0:13]
# Y = dataset[:, 13]

# from utils import encode_numeric_score, to_xy
TRAIN_DATA = "./features/extracted_data/train_preprocessing_output.csv"

data = pd.read_csv(TRAIN_DATA)
# transforming data before training
Y = data['resale_price'].values
X = data.drop('resale_price', axis=1).values


# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=98,
              kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(
    build_fn=wider_model, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=3)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
