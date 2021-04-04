import numpy as np

# param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
#               'eta0': [0.0005, 0.001, 0.01, 0.1, 1, 10],
#               'penalty': ['l1', 'l2'],
#               'learning_rate': ['constant', 'adaptive'],
#               'max_iter': [30000]}

param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
              'eta0': [0.001, 0.01, 0.1, 1],
              'penalty': ['l1', 'l2'],
              'power_t': [-0.35, -0.25, 0, 0.25],
              'learning_rate': ['invscaling'],
              'max_iter': [30000]}

# param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
#               'penalty': ['l1', 'l2'],
#               'learning_rate': ['optimal'],
#               'max_iter': [50000]}