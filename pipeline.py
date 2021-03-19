
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
    TRAIN_DATA = "./features/train_output.csv"
    TRAIN_DATA_OUTPUT = "./features/train_output.csv"
    PREDICTION_OUTPUT = "./results_analysis/results/experiment_output.csv"
    K_FOLD = args.folds

    def run(self):
        start_time = time.time()
        print("Experiment started at ", start_time, "\n")	
        valid_arguments = self.check_arguments()
        if not valid_arguments:
            return

        # Load data
        print("Loading data...")
        train_data = self.load_train_data()
        print(train_data.shape)
        print(train_data.head())

        # Load model
        model = self.load_model()
        
        print(train_data.shape)


        # Hyperparemeter testing loop
        self.load_param_grid()

        
        # Generate k folds 
        (x_fold, y_fold) = self.generate_k_fold(train_data)

        # Run cross validation 
        self.run_cross_validation(x_fold, y_fold, model)

        # Predict test
        test_data = self.load_test_data()
        y_pred = self.predict_test(train_data, test_data, model)
        y_pred.to_csv(self.PREDICTION_OUTPUT)
       
    def load_train_data(self):
        train_data = pd.read_csv(self.TRAIN_DATA)
        return train_data

    def load_model(self):
        model = None
        if (args.model == "linear-regression"):
            from models.linear_regression import LinearRegression
            model = LinearRegression
        return model
        
    def load_param_grid(self):
        if (args.model == "linear-regression"):
            import models.linear_regression_param_grid
        else if (args.model == "random-forest"):
            import models.random_forest
            
    def check_arguments(self):
        if (args.model is None):
            print("Please provide model")
            return False
        if (args.folds is None):
            print("Please provide amount of folds (k)")
            return False
        return True

    def generate_k_fold(self, data):
        print(data.shape)

        x_train = data.drop('resale_price', axis=1)
        y_train = data['resale_price']


        x_folds = []
        y_folds = []
        kf = KFold(n_splits=self.K_FOLD)
        for train_index, test_index in kf.split(data):
            x_folds.append(x_train.iloc[test_index.tolist()])
            y_folds.append(y_train.iloc[test_index.tolist()])
        x_folds = np.array(x_folds)
        y_folds = np.array(y_folds)
        
        return (x_folds, y_folds)

    def run_cross_validation(self, x_fold, y_fold, model):
        for fold in range(0,self.K_FOLD):
            index = [x for i,x in enumerate(fold_iterator) if i!=fold_no] 

            x_train = np.concatenate((x_fold[train_index]), axis=0)
            y_train = np.concatenate((y_fold[train_index]), axis=0)
            x_test = x_fold[fold_no]
            y_test = y_fold[fold_no] 

            model.fit(x_train, y_train)
            pred_test = model.predict(x_test)

            report = classification_report(y_test, pred_test, output_dict=True)

    def predict_test(self, train_data, test_data, model):
        x_train = train_data[~['price']]
        y_train = train_data[['price']]
        x_test = test_data[~['price']]

        model.fit(x_train, y_train)
        y_test = model.predict(x_test)
        return y_test
        
if __name__ == "__main__":
    tqdm.pandas()
    Pipeline().run()
