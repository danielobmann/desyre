import os

from imports.desyre_optimization import DESYRE
from imports.util import Util
from keras.models import load_model

import keras.backend as K
import odl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image

sess = K.get_session()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="paper")
parser.add_argument("-t", "--theta", default=60)
parser.add_argument("-a", "--alpha", default=1e-6)
parser.add_argument("-i", "--iter", default=200)
parser.add_argument("-l", "--learningrate", default=1e-3)
args = vars(parser.parse_args())

path = "models/" + args['path'] + "/"
img_save = "images/" + args['path'] + "/"

if not os.path.exists(img_save):
    os.mkdir(img_save)

# ----------------------------------------
# Set up forward operator using ODL library

cmap = 'gray'
size = 512
n_theta = int(args['theta'])
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

if __name__ == '__main__':

    e = load_model(path + 'encoder.h5', custom_objects=util.custom_objects)
    d = load_model(path + 'decoder.h5', custom_objects=util.custom_objects)

    desyre = DESYRE(encoder=e, decoder=d, operator=Radon, size=size, sess=sess)

    # Set seed for reproducible results
    np.random.seed(0)
    file = np.random.choice(os.listdir("data/test/"))
    phantom = np.asarray(Image.open("data/test/" + file).convert('L'))/255.

    data = Radon(phantom)
    data_noisy = data + np.random.normal(0, 1, data.shape) * 0.0 * np.mean(data)

    # Initialize optimization with FBP-reconstruction
    x0 = util.project(FBP(data_noisy))

    alpha = float(args['alpha'])
    niter = int(args['iter'])
    learning_rate = float(args['learningrate'])
    x_desyre, err = desyre.fista(x0, data_noisy, niter=niter, alpha=alpha, learning_rate=learning_rate)

    fig, axs = plt.subplots(1, 1)
    axs.semilogy(err)
    axs.set_title("Error DESYRE optimization")
    plt.savefig(img_save + "eval_error.pdf")

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
    plt.savefig(img_save + "eval_reconstruction.pdf")

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
    plt.savefig(img_save + "eval_data.pdf")
    plt.clf()
