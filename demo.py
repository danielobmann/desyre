import os

from imports.desyre_optimization import DESYRE
from imports.util import Util
from keras.models import load_model

import keras.backend as K
import odl
import numpy as np
import matplotlib.pyplot as plt
import argparse

sess = K.get_session()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="paper")
args = vars(parser.parse_args())

path = "models/" + args['path'] + "/"

# ----------------------------------------
# Set up forward operator using ODL library

cmap = 'gray'
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
save_path = "images/" + args['path'] + "/"

if __name__ == '__main__':

    e = load_model(path + 'encoder.h5', custom_objects=util.custom_objects)
    d = load_model(path + 'decoder.h5', custom_objects=util.custom_objects)

    desyre = DESYRE(encoder=e, decoder=d, operator=Radon, size=size, sess=sess)

    if not os.path.exists("images/"):
        os.mkdir("images/")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Created image folder for saving images.")

    # Generate a box phantom for demonstration of the algorithm
    phantom = np.zeros((size, size))
    a = 25
    phantom[(size // 2 - a):(size // 2 + a), (size // 2 - a):(size // 2 + a)] = 1

    data = Radon(phantom)
    data_noisy = data + np.random.normal(0, 1, data.shape) * 0.01

    # Initialize optimization with FBP-reconstruction
    x0 = FBP(data_noisy)
    x_desyre, err = desyre.fista(x0, data_noisy, niter=30, alpha=1e-3, learning_rate=1e-3)

    fig, axs = plt.subplots(1, 1)
    axs.semilogy(err)
    axs.set_title("Error DESYRE optimization")
    plt.savefig(save_path + "demo_error.pdf")

    fig, axs = plt.subplots(1, 3)
    im = axs[0].imshow(phantom, cmap=cmap)
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("True phantom")

    im = axs[1].imshow(x0, cmap=cmap)
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title("FBP phantom")

    im = axs[2].imshow(x_desyre, cmap=cmap)
    plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    axs[2].set_title("DESYRE phantom")
    plt.subplots_adjust(wspace=0.8)
    plt.savefig(save_path + "demo_reconstruction.pdf")

    fig, axs = plt.subplots(1, 2)
    im = axs[0].imshow(data, cmap=cmap)
    axs[0].set_aspect(n_s / n_theta)
    axs[0].set_title("True data")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    im = axs[1].imshow(data_noisy, cmap=cmap)
    axs[1].set_aspect(n_s / n_theta)
    axs[1].set_title("Noisy data")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(save_path + "demo_data.pdf")


    def generate_atom(non_zero=5, I=[-1], c=1, idx=None, idy=None):
        xi = [np.zeros(s) for s in desyre.input_shape]
        for i in I:
            j = 0
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

    C = [-3, -2, -1, 1, 2, 3]
    idx, idy = 8, 8
    fig, axs = plt.subplots(4, 6)
    for row, i in enumerate([-1, -2, -3, -4]):
        for col, c in enumerate(C):
            atom = generate_atom(I=[i], c=c, non_zero=1, idx=idx, idy=idy)
            axs[row, col].imshow(atom, cmap=cmap, vmin=0, vmax=0.5)
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title("c=%d" % c)

    plt.subplots_adjust(wspace=-0.1)
    plt.suptitle("Images synthesized from 1 non-zero entry (single levels)")
    plt.savefig(save_path + "demo_atoms.pdf")

    fig, axs = plt.subplots(3, 6)
    for row, i in enumerate([[-3, -1], [-3, -2], [-3, -4]]):
        for col, c in enumerate(C):
            atom = generate_atom(I=i, c=c, non_zero=1, idx=idx, idy=idy)
            axs[row, col].imshow(atom, cmap=cmap, vmin=0, vmax=0.5)
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title("c=%d" % c)

    plt.subplots_adjust(wspace=0.1)
    plt.suptitle("Images synthesized from 1 non-zero entry (combined levels)")
    plt.savefig(save_path + "demo_atoms_combined.pdf")
