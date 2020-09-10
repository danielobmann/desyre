import os

from imports.desyre_optimization import DESYRE
from imports.util import Util
from keras.models import load_model

import keras.backend as K
import odl
import numpy as np
import matplotlib.pyplot as plt

sess = K.get_session()

# ----------------------------------------
# Set up forward operator using ODL library

size = 512
n_theta = 40
n_s = int(1.5 * size)

reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[size, size], dtype='float32')
angle_partition = odl.uniform_partition(0, np.pi * (1 - 1 / (2 * n_theta)), n_theta)
detector_partition = odl.uniform_partition(-1.5, 1.5, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

Radon = odl.tomo.RayTransform(reco_space, geometry)
FBP = odl.tomo.fbp_op(Radon, filter_type='Hann')

# ----------------------------------------
# Set up DESYRE optimization with loaded networks

util = Util()

e = load_model('models/encoder.h5', custom_objects=util.custom_objects)
d = load_model('models/decoder.h5', custom_objects=util.custom_objects)

desyre = DESYRE(encoder=e, decoder=d, operator=Radon, size=size, sess=sess)

if __name__ == '__main__':

    if not os.path.exists("images/"):
        os.mkdir("images/")
        print("Created image folder for saving images.")

    # Generate a box phantom for demonstration of the algorithm
    phantom = np.zeros((size, size))
    phantom[(size // 2 - 50):(size // 2 + 5), (size // 2 - 50):(size // 2 + 5)] = 1

    data = Radon(phantom)
    data_noisy = data + np.random.normal(0, 1, data.shape) * 0.01

    # Initialize optimization with FBP-reconstruction
    x0 = FBP(data_noisy)
    x_desyre, err = desyre.fista(x0, data_noisy, niter=30, alpha=1e-3, learning_rate=1e-3)

    fig, axs = plt.subplots(1, 1)
    axs.semilogy(err)
    axs.set_title("Error DESYRE optimization")
    plt.savefig("images/demo_error.pdf")

    fig, axs = plt.subplots(1, 3)
    im = axs[0].imshow(phantom, cmap='bone')
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("True phantom")

    im = axs[1].imshow(x0, cmap='bone')
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title("FBP phantom")

    im = axs[2].imshow(x_desyre, cmap='bone')
    plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    axs[2].set_title("DESYRE phantom")
    plt.subplots_adjust(wspace=0.8)
    plt.savefig("images/demo_reconstruction.pdf")

    fig, axs = plt.subplots(1, 2)
    im = axs[0].imshow(data, cmap='bone')
    axs[0].set_aspect(n_s / n_theta)
    axs[0].set_title("True data")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    im = axs[1].imshow(data_noisy, cmap='bone')
    axs[1].set_aspect(n_s / n_theta)
    axs[1].set_title("Noisy data")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig("images/demo_data.pdf")


    def generate_atom(non_zero=5, i=-1, c=1, idx=None, idy=None):
        j = 0
        xi = [np.zeros(s) for s in desyre.input_shape]
        temp = np.zeros_like(xi[i])
        x, y = temp.shape[1], temp.shape[2]
        while j < non_zero:
            if idx is None:
                idx = np.random.choice(x)
            if idy is None:
                idy = np.random.choice(y)
            temp[0, idx, idy, 0] = c
            j += 1
        xi[i] = temp
        return d.predict(xi)[0, ..., 0]

    fig, axs = plt.subplots(4, 6)
    for row, i in enumerate([-1, -2, -3, -4]):
        for col, c in enumerate([-3, -2, -1, 1, 2, 3]):
            atom = generate_atom(i=i, c=c, non_zero=1, idx=10, idy=10)
            axs[row, col].imshow(atom, cmap='bone')
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title("c=%d" % c)

    plt.subplots_adjust(wspace=-0.1)
    plt.suptitle("Images synthesized from 1 non-nonzero entry")
    plt.savefig("images/demo_atoms.pdf")
