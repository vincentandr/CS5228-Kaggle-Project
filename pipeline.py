import time
import sys
import numpy as np
import pandas as pd
import argparse
import math

# Import sklearn helpers
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# For convenient vectorized calculations of haversine distance
from haversine import haversine_vector
from sklearn.model_selection import ParameterGrid

# Show progress bar
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="model")
parser.add_argument("-f", "--folds", type=int, help="number of folds for CV")
args = parser.parse_args()

# ----------------Pipeline overview----------------
# Load feature vectors
# Load models
# Generate cross validation folds
# Run K-fold cross validation
# Save checkpoint
# Save results
# ----------------Pipeline overview----------------


class Pipeline:
    # Global variables
    TRAIN_DATA = "./features/extracted_data/train_preprocessing_output.csv"
    TEST_DATA = "./features/extracted_data/test_preprocessing_output.csv"
    PREDICTION_OUTPUT = "./results_analysis/results/experiment_output.csv"
    K_FOLD = args.folds
    param_grid = None

    def run(self):
        start_time = time.time()
        print("Experiment started at ", start_time, "\n")

        # Check if CLI args are valid
        valid_arguments = self.check_arguments()
        if not valid_arguments:
            return

        # Load data
        print("Loading data...")
        train_data = self.load_train_data()

        # Load model
        model = self.load_model()

        # Hyperparemeter testing loop
        self.load_param_grid()

        # # Run hyperparameter search
        # self.run_hyperparameter_search(train_data, model)

        # Run feature selection
        self.run_feature_selection(train_data, model)

        # # # Predict test
        # test_data = self.load_test_data()
        # test_predictions = self.predict_test(train_data, test_data, model)

    def load_train_data(self):
        train_data = pd.read_csv(self.TRAIN_DATA)
        return train_data

    def load_test_data(self):
        train_data = pd.read_csv(self.TEST_DATA)
        return train_data

    def load_model(self):
        """
            Loads model from the models/ folder, or from sklearn
            # Arguments
                None
            # Returns
                None
        """
        print(args.model)
        if (args.model == "linear_regression"):
            from sklearn.linear_model import LinearRegression
            model = LinearRegression
            # print(model)
        if (args.model == "random_forest"):
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor
            # print(model)
        if (args.model == "xg_boost"):
            from xgboost import XGBRegressor
            model = XGBRegressor
            # print(model)
        if (args.model == "ada_boost"):
            from sklearn.ensemble import AdaBoostRegressor
            model = AdaBoostRegressor
            # print(model)
        if (args.model == "sgd_regressor"):
            from sklearn.linear_model import SGDRegressor
            model = SGDRegressors
            # print(model)
        if (args.model == "MLPRegressor"):
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor
            # print(model)
        if (args.model == "decision_trees"):
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor
            # print(model)
        return model

    def load_param_grid(self):
        """
            Loads parameter grid from models/param_grids
            # Arguments
                None
            # Returns
                None
        """
        if (args.model == "linear_regression"):
            from models.param_grids.linear_regression_param_grid import param_grid
            self.param_grid = param_grid
        if (args.model == "random_forest"):
            from models.param_grids.random_forest_param_grid import param_grid
            self.param_grid = param_grid
        if (args.model == "xg_boost"):
            from models.param_grids.xg_boost_param_grid import param_grid
            self.param_grid = param_grid
        if (args.model == "ada_boost"):
            from models.param_grids.ada_boost_param_grid import param_grid
            self.param_grid = param_grid
        if (args.model == "sgd_regressor"):
            from models.param_grids.sgd_regressor_param_grid import param_grid
            self.param_grid = param_grid
        if (args.model == "MLPRegressor"):
            from models.param_grids.MLPRegressor_param_grid import param_grid
            self.param_grid = param_grid
        if (args.model == "decision_trees"):
            from models.param_grids.decision_trees_param_grid import param_grid
            self.param_grid = param_grid

    def check_arguments(self):
        """
            Check command line arguments
            # Arguments
                None
            # Returns
                None
        """
        if (args.model is None):
            print("Please provide model")
            return False
        if (args.folds is None):
            print("Please provide amount of folds (k)")
            return False
        return True

    def generate_k_fold(self, data):
        """
            Generate K_Fold (this is in a function because initially this process is done manually, not using sklearn)
            # Arguments
                data: data that k_fold indices are based on
            # Returns
                kf: k_fold indices
        """
        kf = KFold(n_splits=self.K_FOLD)
        return kf.split(data)

    def run_hyperparameter_search(self, data,  model):
        """
            This function generates the self.best_parameter value, after doing grid search on param_grid
            # Arguments
                data: that will be used during hyperparameter search
                model: the type of model used
            # Returns
                None
        """
        # transforming data before training
        train_labels = data['resale_price']
        train_features = data.drop('resale_price', axis=1)

        best_rmse = math.inf
        # generate iteratble param combinations from param_Grid
        param_combinations = list(ParameterGrid(self.param_grid))
        for current_paramters in param_combinations:

            current_fold = 1  # Current fold to track progress during k_fold
            total_rmse = 0  # Total rmse so far

            print("Current param: ", current_paramters)
            for train_index, test_index in self.generate_k_fold(data):

                # Getting the current fold's data
                X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
                y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

                # Training the model
                current_model = model(**current_paramters)
                current_model.fit(X_train, y_train)
                pred_test = current_model.predict(X_test)

                # Calculating RMSE
                score = mean_squared_error(y_test, pred_test, squared=False)
                print("Fold: ", current_fold, ", RMSE: ", score)
                current_fold = current_fold + 1
                total_rmse = total_rmse + score

                self.best_parameters = current_paramters

            # Comparing the RMSE of current parameters to the best so far
            avg_rmse = total_rmse / self.K_FOLD
            if (avg_rmse < best_rmse):
                best_rmse = avg_rmse
                self.best_parameters = current_paramters
            print("Average RMSE: ", total_rmse/self.K_FOLD)

    def run_feature_selection(self, data,  model):
        """
            This function generates the self.best_no_of_features value, after doing feature selection
            # Arguments
                data: that will be used during hyperparameter search
                model: the type of model used
            # Returns
                None
        """
        # transforming data before training and prediction
        train_labels = data['resale_price']
        train_features = data.drop('resale_price', axis=1)

        best_rmse = math.inf
        for current_k in range(1, len(data.columns[1:]) + 1):
            train_new_features = SelectKBest(
                f_classif, k=current_k).fit_transform(train_features, train_labels)

            # Split train test data
            X_train, X_test, y_train, y_test = train_test_split(
                train_new_features, train_labels, random_state=42, test_size=0.1)

            # Training the model
            current_model = model()
            current_model.fit(X_train, y_train)

            # Predict
            predictions = current_model.predict(X_test)
            current_rmse = mean_squared_error(
                predictions, y_test, squared=False)

            print("K: ", current_k, ", RMSE: ", current_rmse)

            if (current_rmse < best_rmse):
                best_rmse = current_rmse
                self.best_no_of_features = current_k
        print("Best RMSE : ",  best_rmse, " K: ", self.best_no_of_features)

    def predict_test(self, train_data, test_data, model):
        """
            Predicting final test data, and saving it to csv
            # Arguments
                train_data: data used to train our final model
                test_data: dta that will be predicted
                model: model that will be used
            # Returns
                None
        """
        print("Predicting using : ", self.best_parameters)

        # transforming data before training and prediction
        train_labels = train_data['resale_price']
        train_features = train_data.drop('resale_price', axis=1)

        test_data['block'] = test_data['block'].str.replace(
            '[A-Za-z]', '', regex=True)
        test_data['block'] = test_data['block'].apply(int)

        # training the model using best parameters
        current_model = model(**self.best_parameters)
        current_model.fit(train_features, train_labels)

        # actual prediction using model
        test_prediction = current_model.predict(test_data)

        # saving data to csv
        result = pd.DataFrame(
            {'Id': np.arange(len(test_predictions)), 'Predicted': test_predictions})
        result.to_csv(self.PREDICTION_OUTPUT, index=False)


if __name__ == "__main__":
    tqdm.pandas()
    Pipeline().run()
