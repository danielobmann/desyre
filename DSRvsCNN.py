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

sess = K.get_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

co = CustomObjects(sess)
img_folder = 'images/versus/'

try:
    os.mkdir(img_folder)
except:
    pass

e = load_model('networks/tightframe/models/good/encoder.h5', custom_objects=co.custom_objects)
d = load_model('networks/tightframe/models/good/decoder.h5', custom_objects=co.custom_objects)
pp = load_model('networks/postprocessing/models/postprocessing.h5', custom_objects=co.custom_objects)

img_height, img_width = 512, 512
n_theta = 30
n_s = int(1.5 * img_height)

reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[img_height, img_width], dtype='float32')
angle_partition = odl.uniform_partition(np.pi / (2 * n_theta), np.pi * (1 - 1 / (2 * n_theta)), n_theta)
detector_partition = odl.uniform_partition(-1.5, 1.5, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
FBP = odl.tomo.fbp_op(ray_trafo, filter_type='Hann')

# Regularizer initialization
snett = SNETT(e, d, ray_trafo, img_height, img_width, sess)

ph = np.asarray(Image.open('data/versus_4.png').convert('L'))/255.
data = ray_trafo(ph)
fbp = FBP(data)

xlim = [250, 300]
ylim = [350, 400]

textloc = [0.01, 0.94]
fsize = 20
zoom = 2.5
cmap = 'gray'

niter = 2000 # Reduce number of iterations?
lr = 10**(-3)

xdsr, err_dsr, _ = snett.fista_optimize(co.project(fbp), data, alpha=10**(-6), learning_rate=lr, niter=niter)

plt.semilogy(err_dsr)
plt.savefig(img_folder + 'ERROR.pdf', format='pdf')
plt.clf()

xpost = np.asarray(fbp-pp.predict(co.check_dim(fbp)).reshape((img_height, img_width)))

REC = [xdsr, xpost, FBP(data), ph]
NAME = ['DESYRE', 'Post-processing', 'FBP', 'True']

for rec, name in zip(REC, NAME):
    co.zoomed_plot(rec, xlim=xlim, ylim=ylim, textloc=textloc, text=name, cmap=cmap, zoom=zoom)
    print(co.NMSE(ph, rec))
    print(co.PSNR(ph, rec))
    plt.savefig(img_folder + name + '_versus.pdf', format='pdf')
    plt.clf()




