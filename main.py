import pdb
import time
import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import os
import pickle
import argparse
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import Optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--maxiter', type=int, default=2000)
    parser.add_argument('-ntr', '--num-train-samples', type=int, default=10000)
    parser.add_argument('-nte', '--num-test-samples', type=int, default=100)
    parser.add_argument('-n', '--network', type=str, default='pinn')
    parser.add_argument('-l', '--loss', type=str, default='l2')
    parser.add_argument('-gi', '--gradient-interval', type=int, default=100)
    parser.add_argument('--gt-path', type=str, default='data/pinn.pkl')
    return parser.parse_known_args()[0]


def uv(network, xy):
    """
    Compute flow velocities (u, v) for the network with output (psi, p).

    Args:
        xy: network input variables as ndarray.

    Returns:
        (u, v) as ndarray.
    """

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        psi_p = network(xy)
    psi_p_j = g.batch_jacobian(psi_p, xy)
    u =  psi_p_j[..., 0, 1]
    v = -psi_p_j[..., 0, 0]
    return u.numpy(), v.numpy()

def contour(grid, x, y, z, title, levels=50):
    """
    Contour plot.

    Args:
        grid: plot position.
        x: x-array.
        y: y-array.
        z: z-array.
        title: title string.
        levels: number of contour lines.
    """

    # get the value range
    vmin = -2e-1
    vmax = 2e-1
    if (title == 'psi'):
        vmax = 1.2e-1
        vmin = -1e-1
    if (title == 'p'):
        vmax = 6.1e-1
        vmin = -5e-1
    if (title == 'u'):
        vmax = 1.1e+0
        vmin = -2e-1
    if (title == 'v'):
        vmax = 2.1e-1
        vmin = -2e-1
    if (title == 'dpsi'):
        vmax = 1.1e-2
        vmin = 0.0
    if (title == 'dp'):
        vmax = 4.1e-1
        vmin = 0.0
    if (title == 'du'):
        vmax = 1.1e-1
        vmin = 0.0
    if (title == 'dv'):
        vmax = 8.1e-2
        vmin = 0.0
    
    # plot a contour
    plt.subplot(grid)
    print(title, vmin, vmax)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels, vmin=vmin, vmax=vmax)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, vmin=vmin, vmax=vmax)
    plt.title(title)
    m = plt.cm.ScalarMappable(cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax)) 
    m.set_array(z) 
    m.set_clim(vmin, vmax) 
    cbar = plt.colorbar(m, pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)


if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model
    for the cavity flow governed by the steady Navier-Stokes equation.
    """

    args = parse_args()
    # number of training samples
    num_train_samples = args.num_train_samples
    # number of test samples
    num_test_samples = args.num_test_samples

    # inlet flow velocity
    u0 = 1
    # density
    rho = 1
    # viscosity
    nu = 0.01

    # build a core network model
    network = Network().build()
    network.summary()
    # build a PINN model
    model = PINN(network, rho=rho, nu=nu).build()

    # create training input
    xy_eqn = np.random.rand(num_train_samples, 2)
    xy_ub = np.random.rand(num_train_samples//2, 2)  # top-bottom boundaries
    xy_ub[..., 1] = np.round(xy_ub[..., 1])          # y-position is 0 or 1
    xy_lr = np.random.rand(num_train_samples//2, 2)  # left-right boundaries
    xy_lr[..., 0] = np.round(xy_lr[..., 0])          # x-position is 0 or 1
    xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr]))
    x_train = [xy_eqn, xy_bnd]

    # create training output
    zeros = np.zeros((num_train_samples, 2))
    uv_bnd = np.zeros((num_train_samples, 2))
    uv_bnd[..., 0] = u0 * np.floor(xy_bnd[..., 1])
    y_train = [zeros, zeros, uv_bnd]

    # train the model using L-BFGS-B algorithm
    optimizer = Optimizer(model=model, x_train=x_train, y_train=y_train, dict_params=args.__dict__)
    optimizer.fit()

    # create meshgrid coordinates (x, y) for test plots
    x = np.linspace(0, 1, num_test_samples)
    y = np.linspace(0, 1, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    # predict (psi, p)
    psi_p = network.predict(xy, batch_size=len(xy))
    psi, p = [ psi_p[..., i].reshape(x.shape) for i in range(psi_p.shape[-1]) ]
    # compute (u, v)
    u, v = uv(network, xy)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)
    if os.path.isfile(args.gt_path):
        with open(args.gt_path, 'rb') as f:
            data = pickle.load(f)
        x_gt, y_gt, psi_gt, p_gt, u_gt, v_gt = data
        fig = plt.figure(figsize=(6, 5))
        gs = GridSpec(2, 2)
        contour(gs[0, 0], x, y, np.abs(psi - psi_gt), 'dpsi')
        contour(gs[0, 1], x, y, np.abs(p - p_gt), 'dp')
        contour(gs[1, 0], x, y, np.abs(u - u_gt), 'du')
        contour(gs[1, 1], x, y, np.abs(v - v_gt), 'dv')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', list(args.__dict__.values())[:-1].__str__() + str(time.time()) +
                                 '_error.png'))
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(6, 5))
        gs = GridSpec(2, 2)
        contour(gs[0, 0], x, y, psi, 'psi')
        contour(gs[0, 1], x, y, p, 'p')
        contour(gs[1, 0], x, y, u, 'u')
        contour(gs[1, 1], x, y, v, 'v')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', list(args.__dict__.values())[:-1].__str__() + str(time.time()) + '.png'))
        plt.show()
        plt.close()
    else:
        # plot test results
        fig = plt.figure(figsize=(6, 5))
        gs = GridSpec(2, 2)
        contour(gs[0, 0], x, y, psi, 'psi')
        contour(gs[0, 1], x, y, p, 'p')
        contour(gs[1, 0], x, y, u, 'u')
        contour(gs[1, 1], x, y, v, 'v')
        data = [x, y, psi, p, u, v]
        with open(args.gt_path, 'wb') as f:
            pickle.dump(data, f)
        plt.tight_layout()
        plt.savefig(os.path.join('figures', list(args.__dict__.values())[:-1].__str__() + str(time.time()) + '.png'))
        plt.show()
        plt.close()
