from imports.customobjects import *
from imports.snett_efficient import *
import keras.backend as K
from keras.models import load_model
import numpy as np
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

graph = tf.get_default_graph()
sess = K.get_session()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess = K.get_session()
co = CustomObjects(sess)

nettype = 'tightframe/models/'
e = load_model('networks/' + nettype + 'good/encoder.h5', custom_objects=co.custom_objects)
d = load_model('networks/' + nettype + 'good/decoder.h5', custom_objects=co.custom_objects)


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
datadirectory = "data/"
snett = SNETT(e, d, ray_trafo, img_height, img_width, sess)

niterL1 = 100
lrL1 = 10 ** (-3)


def project(x):
    return np.clip(x, 0, 1)

NAMES = ['L506_2', 'L506_20', 'L506_35'] #try 506_50 or 333_1 or 333_75

for name in NAMES:
    file = datadirectory + 'reconstruct/' + name + '.png'
    ph = np.asarray(Image.open(file).convert('L'))/255.
    data = ray_trafo(ph)
    fbp = FBP(data)

    xfista, errfista = snett.fista_optimize(project(FBP(data)), data, alpha=10**(-4), learning_rate=lrL1, niter=niterL1)

    plt.semilogy(errfista, label='FISTA')
    plt.legend()
    plt.savefig('images/Error'+name+'.pdf', format='pdf')
    plt.clf()

    plt.imshow(xfista, cmap='gray', vmin=0.0, vmax=1.0)
    plt.savefig('images/reconstructionFISTA' + name + '.pdf', format='pdf')
    plt.clf()

    print(name)
    print("PSNR (FISTA)", co.PSNR(ph, xfista))
    print("NMSE (FISTA)", co.NMSE(ph, xfista))