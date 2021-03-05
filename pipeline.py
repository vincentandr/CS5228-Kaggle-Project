
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

import numpy as np
import pandas as pd

# Import sklearn helpers
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold

# For convenient vectorized calculations of haversine distance
from haversine import haversine_vector

# Show progress bar
from tqdm import tqdm


class Pipeline:
    TRAIN_DATA = "./features/feature_vectors/train.csv"
    TRAIN_DATA_OUTPUT = "./features/feature_vectors/train_output.csv"
    PREDICTION_OUTPUT = "./results_analysis/results/experiment_output.csv"

    K_FOLD = 5

    def run(self):
        start_time = time.time()
        print("Experiment started at ", start_time.strftime("%d/%m/%Y %H:%M:%S"), "\n")	

        # Load data
        print("Loading data...")
        train_data = self.load_train_data()
        print(train_data.shape)
        print(train_data.head())

        # Generate k folds
        (x_fold, y_fold) = self.generate_k_fold(train_data)

        # Run cross validation 
        self.run_cross_validation(x_fold, y_fold, model)

        # Predict test
        test_data = self.load_test_data()
        y_pred = self.predict_test(train_data, test_data, model)
        y_pred.to_csv(self.PREDICTION_OUTPUT)
       
    def load_train_data(self, data):
        train_data = pd.read_csv(self.TRAIN_DATA)
        return train_data

    def generate_k_fold(self, data):
        x_folds = []
        y_folds = []
        skf = StratifiedKFold(n_splits=no_fold)
        for train_index, test_index in skf.split(file_code_name, file_code_labels):
            x_folds.append(scaler.transform(flatten(x_train_original[test_index])))
            y_folds.append(flatten(y_train_original[test_index]))
        x_folds = np.array(x_folds)
        y_folds = np.array(y_folds)
        return (x_fold, y_fold)

    def run_cross_validation(self, x_fold, y_fold, model):
        for fold in range(0,self.K_FOLD):
            index = [x for i,x in enumerate(fold_iterator) if i!=fold_no] 

            x_train = np.concatenate((x_fold[train_index]), axis=0)
            y_train = np.concatenate((x_fold[train_index]), axis=0)
            x_test = y_fold[fold_no]
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
