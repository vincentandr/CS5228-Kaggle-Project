
# ----------------Pipeline overview----------------
# Load feature vectors
# Load models
# Generate cross validation folds
# Run K-fold cross validation
    # Save checkpoint
# Save results
# ----------------Pipeline overview----------------





# # Log start of experiment
# start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
# print("Experiment "+ experiment_name +" started at ", exp_start_time, "\n")	

# # save experiment
# save_pickle(results, './results/'+experiment_name+'_results_bootstrap.sav')    

# # Log end of experiment
# print("Total time elapsed : " + str(math.floor(total_time_elapsed/60)) + "m " + str(round(total_time_elapsed % 60)) + "s")
# print("Experiment started at ", exp_start_time)	
# print("Experiment ended at ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))	






import time
import sys
import numpy as np
import pandas as pd
import argparse


# Import sklearn helpers
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.metrics import classification_report,mean_squared_error
# For convenient vectorized calculations of haversine distance
from haversine import haversine_vector

# Show progress bar
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="model")
parser.add_argument("-f", "--folds", type=int, help="number of folds for CV")
args = parser.parse_args()

class Pipeline:
    TRAIN_DATA = "./features/extracted_data/train_standardizedohe_output.csv"
    TRAIN_DATA_OUTPUT = "./features/extracted_data/train_standardizedohe_output.csv"
    PREDICTION_OUTPUT = "./results_analysis/results/experiment_output.csv"
    K_FOLD = args.folds
    param_grid = None

    def run(self):
        start_time = time.time()
        print("Experiment started at ", start_time, "\n")	
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

        # Generate k folds 
        folds = self.generate_k_fold(train_data)

        # Run hyperparameter search 
        self.run_hyperparameter_search(folds, train_data, model)


        # # Predict test
        # test_data = self.load_test_data()
        # y_pred = self.predict_test(train_data, test_data, model)
        # y_pred.to_csv(self.PREDICTION_OUTPUT)
       
    def load_train_data(self):
        train_data = pd.read_csv(self.TRAIN_DATA)
        return train_data

    def load_model(self):
        model = None
        print(args.model)
        # if (args.model == "linear_regression"):
        #     from models.linear_regression import LinearRegression
        #     model = LinearRegression()
        #     print(model)
        if (args.model == "linear_regression"):
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            # print(model)
        if (args.model == "random_forest"):
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(verbose = 2)
            # print(model)
        return model
        
    def load_param_grid(self):
        if (args.model == "linear_regression"):
            from models.linear_regression_param_grid import param_grid
            self.param_grid = param_grid
        if (args.model == "random_forest"):
            from models.random_forest_param_grid import param_grid
            self.param_grid = param_grid
            
    def check_arguments(self):
        if (args.model is None):
            print("Please provide model")
            return False
        if (args.folds is None):
            print("Please provide amount of folds (k)")
            return False
        return True

    def generate_k_fold(self, data):
        kf = KFold(n_splits=self.K_FOLD)
        return kf.split(data)

    def run_hyperparameter_search(self, folds, data,  model):
        train_labels = data['resale_price'].values
        train_features = data.drop('resale_price', axis=1).values
        print(train_labels.shape)
        print(train_features.shape)

        for train_index, test_index in folds:
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_features[train_index], train_features[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            
            # X_train, X_test = train_features.iloc(train_index), train_features.iloc(test_index)
            # y_train, y_test = train_labels.iloc(train_index), train_labels.iloc(test_index)
            model.fit(X_train, y_train)
            pred_test = model.predict(X_test)

            score = mean_squared_error(y_test, pred_test, squared=False)
            print(score)

    def predict_test(self, train_data, test_data, model):
        x_train = train_data[~['price']]
        y_train = train_data[['price']]
        x_test = test_data[~['price']]

        model.fit(X_train=x_train, y_train=y_train)
        y_test = model.predict(x_test)
        return y_test
        
if __name__ == "__main__":
    tqdm.pandas()
    Pipeline().run()
