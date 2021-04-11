import numpy as np
from sklearn.tree import DecisionTreeRegressor

param_grid = {
	"base_estimator"	: [DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=7), DecisionTreeRegressor(max_depth=10)],
	"n_estimators" 		: [200, 500, 1000],
	"learning_rate"     : [1.0],
	"random_state"		: [42],
}