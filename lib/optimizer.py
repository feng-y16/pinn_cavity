import pdb

import scipy.optimize
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Optimizer:
    """
    Optimize the keras network model using adam algorithm.

    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10.0 for extremely high accuracy.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: log metrics
        progbar: progress bar
    """

    def __init__(self, model, x_train, y_train, maxiter=5000, dict_params=None):
        """
        Args:
            model: optimization target model.
            samples: training samples.
            maxiter: maximum number of iterations.
        """

        # set attributes
        self.model = model
        self.x_train = [tf.Variable(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]
        self.maxiter = maxiter
        self.metrics = ['loss']
        for key in dict_params:
            self.__setattr__(key, dict_params[key])

    def regenerate_data(self, grads_x):
        tx_eqn, tx_ini, tx_bnd = self.x_train
        self.x_train[0] = tf.Variable(tf.concat((tx_eqn,
                                                 tf.add(tx_eqn, grads_x[0] * 0.01)), axis=0))
        num_train_samples = tx_ini.shape[0]
        new_tx_ini = 2 * np.random.rand(num_train_samples, 2) - 1  # x_ini = -1 ~ +1
        new_tx_ini[..., 0] = 0  # t_ini =  0
        self.x_train[1] = tf.Variable(tf.concat((tx_ini, tf.constant(new_tx_ini, dtype=tf.float32)), axis=0))

        new_tx_bnd = np.random.rand(num_train_samples, 2)  # t_bnd =  0 ~ +1
        new_tx_bnd[..., 1] = 2 * np.round(new_tx_bnd[..., 1]) - 1  # x_bnd = -1 or +1
        self.x_train[2] = tf.Variable(tf.concat((tx_bnd, tf.constant(new_tx_bnd, dtype=tf.float32)), axis=0))

        tx_ini = self.x_train[1].numpy()
        num_train_samples = tx_ini.shape[0]
        u_eqn = np.zeros((num_train_samples, 1))  # u_eqn = 0
        if self.network == 'pinn':
            u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])  # u_ini = -sin(pi*x_ini)
        else:
            u_ini = -np.pi * np.cos(-np.pi * tx_ini[..., 1, np.newaxis])  # u_ini = -sin(pi*x_ini)
        u_bnd = np.zeros((num_train_samples, 1))  # u_bnd = 0
        y_train = [u_eqn, u_ini, u_bnd]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]

    @tf.function
    def evaluate(self, x, y, p=2):
        """
        Evaluate loss and gradients for weights as tf.Tensor.

        Args:
            x: input data.
            y: input label.
            p: the norm to be used.

        Returns:
            loss and gradients for weights as tf.Tensor.
        """
        with tf.GradientTape() as g:
            u = self.model(x)
            loss = tf.reduce_mean(tf.keras.losses.mae(u[: 3], y) ** p)
            if len(u) > 3:
                loss += tf.reduce_mean(tf.keras.losses.mae(u[3], tf.zeros(u[3].shape)) ** p)
        grads, grads_x = g.gradient(loss, [self.model.trainable_variables, x])
        return loss, grads, grads_x

    def fit(self):
        """
        Train the model using the adam algorithm.
        """
        optimizer_nn = tf.keras.optimizers.Adam(learning_rate=0.001)
        pbar = tqdm(total=self.maxiter)
        for i in range(self.maxiter):
            p = 2
            if self.loss == 'lp':
                p = int(1 + 1 / (1.1 - i * 1.0 / self.maxiter))
            loss, grads, grads_x = self.evaluate(self.x_train, self.y_train, p)
            if self.loss == 'ag' and (i + 1) % self.gradient_interval == 0:
                self.regenerate_data(grads_x)
            optimizer_nn.apply_gradients(zip(grads, self.model.trainable_variables))
            pbar.set_postfix({'loss': '{:.5f}'.format(loss.numpy())})
            pbar.update()
        pbar.close()
