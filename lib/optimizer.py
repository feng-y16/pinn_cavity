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
        xy_eqn, xy_bnd = self.x_train
        num_train_samples = xy_eqn.shape[0]
        self.x_train[0] = tf.Variable(tf.concat((xy_eqn,
                                                 tf.add(xy_eqn, grads_x[0] * 0.01)), axis=0))
        new_xy_ub = np.random.rand(num_train_samples // 2, 2)  # top-bottom boundaries
        new_xy_ub[..., 1] = np.round(new_xy_ub[..., 1])  # y-position is 0 or 1
        new_xy_lr = np.random.rand(num_train_samples // 2, 2)  # left-right boundaries
        new_xy_lr[..., 0] = np.round(new_xy_lr[..., 0])  # x-position is 0 or 1
        new_xy_bnd = np.random.permutation(np.concatenate([new_xy_ub, new_xy_lr]))
        self.x_train[1] = tf.Variable(tf.concat((xy_bnd, tf.constant(new_xy_bnd, dtype=tf.float32)), axis=0))

        num_train_samples *= 2
        u0 = 1
        # create training output
        zeros = np.zeros((num_train_samples, 2))
        uv_bnd = np.zeros((num_train_samples, 2))
        uv_bnd[..., 0] = u0 * np.floor(self.x_train[1].numpy()[..., 1])
        y_train = [zeros, zeros, uv_bnd]
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
