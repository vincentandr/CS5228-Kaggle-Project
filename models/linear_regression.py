from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np


class LinearRegression:
    rng = np.random

    W = tf.Variable(rng.randn(), name="weight")
    B = tf.Variable(rng.randn(), name="bias")
    # Linear regression (Wx + b).
    def linear_regression( x):
            return W * x + B

    def fit(self, train_x, train_y):
        # Mean square error.
        def mean_square(y_pred, y_true):
            return tf.reduce_mean(tf.square(y_pred - y_true))

        # Stochastic Gradient Descent Optimizer.
        optimizer = tf.optimizers.SGD(learning_rate)
        
        # Linear regression (Wx + b).
        def linear_regression(x):
            return self.W * x + self.B
            
        # Optimization process. 
        def run_optimization():
            # Wrap computation inside a GradientTape for automatic differentiation.
            with tf.GradientTape() as g:
                pred = linear_regression(X)
                loss = mean_square(pred, Y)

            # Compute gradients.
            gradients = g.gradient(loss, [self.W, self.B])
            
            # Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, [self.W, self.B]))

        # Run training for the given number of steps.
        for step in range(1, training_steps + 1):
            # Run the optimization to update W and b values.
            run_optimization()
            
            if step % display_step == 0:
                pred = linear_regression(X)
                loss = mean_square(pred, Y)
                print("step: %i, loss: %f, W: %f, B: %f" % (step, loss,  self.W.numpy(), self.B.numpy()))

    def predict(self, test_x):
        # Linear regression (Wx + b).
        def linear_regression(x):
            return self.W * x + self.B

        test_y = linear_regression(test_x)
        return test_y
