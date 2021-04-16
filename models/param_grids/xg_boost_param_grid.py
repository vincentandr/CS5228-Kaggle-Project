import numpy as np

param_grid = {
    "tree_method": ['gpu_hist'],
    "verbosity": [1],
    "n_estimators": [200, 500, 1000, 1500],
    "random_state": [42],
    "max_depth": [8, 10, 12],
    "learning_rate": [0.05],
    "booster": ['gbtree'],
}
