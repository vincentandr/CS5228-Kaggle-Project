from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
rng = np.random

# Hyper parameters.
learning_rate = 0.01
training_steps = 1000
display_step = 50

# Training Data.
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])
              # Weight and Bias, initialized randomly.

# Training Data.
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])
lrg = LinearRegression()
lrg.fit(X,Y)
pred = lrg.predict(X)

# Mean square error.
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))
loss = mean_square(pred, Y)
print("final loss")
print(loss)


class LinearRegression:
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
