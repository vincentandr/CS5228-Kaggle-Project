
# ----------------Pipeline overview----------------
# Load feature vectors
# Load models
# Generate cross validation folds
# Run K-fold cross validation
    # Save checkpoint
# Save results
# ----------------Pipeline overview----------------





# Log start of experiment
start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Experiment "+ experiment_name +" started at ", exp_start_time, "\n")	

# save experiment
save_pickle(results, './results/'+experiment_name+'_results_bootstrap.sav')    

# Log end of experiment
print("Total time elapsed : " + str(math.floor(total_time_elapsed/60)) + "m " + str(round(total_time_elapsed % 60)) + "s")
print("Experiment started at ", exp_start_time)	
print("Experiment ended at ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))	






import time

import numpy as np
import pandas as pd

# For convenient vectorized calculations of haversine distance
from haversine import haversine_vector

# Show progress bar
from tqdm import tqdm


class Pipeline:
    TRAIN_DATA = "train.csv"
    TEST_DATA = "test.csv"

    TRAIN_DATA_OUTPUT = "./features/feature_vectors/train_output.csv"

    def run(self):
        start_time = time.time()
        print("Experiment "+ experiment_name +" started at ", start_time.strftime("%d/%m/%Y %H:%M:%S"), "\n")	

        # Load data
        print("Loading data...")

        (train_data, test_data) = self.load_train_test_data()
        self.load_auxillary_data()

        print(train_data.shape)
        print(train_data.head())

if __name__ == "__main__":
    tqdm.pandas()
    FeatureExtraction().run()
