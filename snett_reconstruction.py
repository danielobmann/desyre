from imports.customobjects import *
from imports.snett_efficient import *
import keras.backend as K
from keras.models import load_model
import numpy as np
import odl
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


graph = tf.get_default_graph()
sess = K.get_session()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess = K.get_session()
co = CustomObjects(sess)

nettype = 'tightframe/models/'
e = load_model('networks/' + nettype + 'encoder.h5', custom_objects=co.custom_objects) # Use good/ instead
d = load_model('networks/' + nettype + 'decoder.h5', custom_objects=co.custom_objects) # Use good/ instead

# Hyperparameters
img_height, img_width = 512, 512
n_theta = 60
n_s = int(1.5 * img_height)

reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[img_height, img_width], dtype='float32')
angle_partition = odl.uniform_partition(np.pi / (2 * n_theta), np.pi * (1 - 1 / (2 * n_theta)), n_theta)
detector_partition = odl.uniform_partition(-1.5, 1.5, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
FBP = odl.tomo.fbp_op(ray_trafo, filter_type='Hann')

# Regularizer initialization
snett = SNETT(e, d, ray_trafo, img_height, img_width, sess)

niterL1 = 2000
alphaL1 = 10 ** (-6)
lrL1 = 10 ** (-4)

xlim = [200, 300]
ylim = [200, 250]
textloc = [0.65, 0.02]
fsize = 20
zoom = 2
cmap = 'gray'

datadirectory = "data/reconstruct/"
imgdirectory = "images/"
savedir = imgdirectory + 'SNETT_FISTA_NEW'

try:
    os.mkdir(savedir)
except:
    pass

nmse_list = []
psnr_list = []


def project(x):
    return np.clip(x, 0, 1)


def plot(x, text, savepath, name):

    co.plot(x, text=[text], save=savepath + name + '.pdf', cmap=cmap, colorbar=False)
    plt.clf()

    co.zoomed_plot(x, xlim, ylim, zoom=zoom, text=text, textloc=textloc, cmap=cmap)
    plt.savefig(savepath + 'Zoomplot' + name + '.pdf', format='pdf')
    plt.clf()

    pass


def run_optimization(ground_truth, alpha, lr, niter, savepath):

    data = ray_trafo(ground_truth)
    x0 = project(FBP(data))

    xsnett, errsnett, _ = snett.fista_optimize(x0, data, alpha=alpha, learning_rate=lr, niter=niter)

    xsnett = project(xsnett)
    co.error_plot(errsnett, 'SNettError', savepath)

    psnr = co.PSNR(ground_truth, xsnett)
    nmse = co.NMSE(ground_truth, xsnett)

    nmse_list.append(nmse)
    psnr_list.append(psnr)

    plot(xsnett, '{:.2e}'.format(nmse), savepath=savepath, name='SNETT')
    plot(ground_truth, " ", savepath=savepath, name='GroundTruth')
    pass


for filename in os.listdir(datadirectory):
    if filename.endswith(".png"):

        ph = np.asarray(Image.open(datadirectory + filename).convert('L')) / 255.
        folder = savedir + '/' + filename[:-4]
        os.mkdir(folder)
        savepath = folder + '/'

        data = ray_trafo(ph)

        fig, ax = plt.subplots()
        ax.imshow(data, cmap=cmap, aspect=n_s // n_theta)
        ax.axis('off')
        plt.savefig(savepath + 'data.pdf', format='pdf')
        plt.clf()

        run_optimization(ph, alpha=alphaL1, lr=lrL1, niter=niterL1, savepath=savepath)


print("########### PSNR ##############")
print(psnr_list)

print(np.mean(psnr_list))
print(np.std(psnr_list))

print("########### NMSE ##############")
print(nmse_list)

print(np.mean(nmse_list))
print(np.std(nmse_list))
