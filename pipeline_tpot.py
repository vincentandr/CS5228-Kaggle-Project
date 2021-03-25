from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


TRAIN_DATA_OUTPUT = "./features/extracted_data/train_preprocessing_output.csv"
train_data = pd.read_csv(TRAIN_DATA_OUTPUT)
train_features = train_data.drop('resale_price', axis=1)
train_labels = train_data['resale_price']


housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels,
                                                    train_size=0.8, test_size=0.2, random_state=42)

tpot = TPOTRegressor(generations=6, population_size=30, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')