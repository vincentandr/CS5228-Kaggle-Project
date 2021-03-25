from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

# Hyper parameters.
learning_rate = 0.01
training_steps = 1000
display_step = 50


class LinearRegression:
    W = None
    B = None

    def fit(self, X_train, y_train):

        W = tf.Variable(np.random.normal(), name='W')
        B = tf.Variable(np.random.normal(), name='b')

        # Mean square error.
        def mean_square(y_pred, y_true):
            return tf.reduce_mean(tf.square(y_pred - y_true))

        # Stochastic Gradient Descent Optimizer.
        optimizer = tf.optimizers.SGD(learning_rate)

        # Run training for the given number of steps.
        for step in range(1, training_steps + 1):
            # Run the optimization to update W and b values.
            with tf.GradientTape() as g:
                print(W.shape)
                print(X_train.shape)
                print(B.shape)
                pred = tf.add(tf.multiply(W, X_train), B)
                print(pred.shape)
                print(y_train.shape)
               
                loss = mean_square(pred, y_train)

            # Compute gradients.
            gradients = g.gradient(loss, [self.W, self.B])
            
            # Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, [self.W, self.B]))
            
            if step % display_step == 0:
                pred = tf.add(tf.multiply(W, X_train), B)
                loss = mean_square(pred, y_train)
                print("step: %i, loss: %f, W: %f, B: %f" % (step, loss,  self.W.numpy(), self.B.numpy()))

    def predict(self, X_test):
        # Linear regression (Wx + b).
        pred = tf.add(tf.multiply(W, X_test), B)
        return pred
