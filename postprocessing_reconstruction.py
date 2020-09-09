from imports.customobjects import *

import keras.backend as K
from keras.models import load_model

import numpy as np
import odl
import matplotlib.pyplot as plt
from PIL import Image

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess = K.get_session()
co = CustomObjects(sess)

nettype_post = 'postprocessing'

post_processing_network = load_model('networks/' + nettype_post + '/models/postprocessing.h5', custom_objects=co.custom_objects)

######## Setting up inverse problem ############
img_height, img_width = 512, 512
n_theta = 60
n_s = int(1.5 * img_height)

reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[img_height, img_width], dtype='float32')
angle_partition = odl.uniform_partition(np.pi / (2 * n_theta), np.pi * (1 - 1 / (2 * n_theta)), n_theta)
detector_partition = odl.uniform_partition(-1.5, 1.5, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
FBP = odl.tomo.fbp_op(ray_trafo, filter_type='Hann')


# Plotting parameters
xlim = [200, 300]
ylim = [200, 250]
textloc = [0.65, 0.02]
fsize = 20
zoom = 2
cmap = 'gray'


def plot(x, text, savepath, name):

    co.plot(x, text=[text], save=savepath + name + '.pdf', cmap=cmap, colorbar=False)
    plt.clf()

    co.zoomed_plot(x, xlim, ylim, zoom=zoom, text=text, textloc=textloc, cmap=cmap)
    plt.savefig(savepath + 'Zoomplot' + name + '.pdf', format='pdf')
    plt.clf()

    pass


datadirectory = "data/reconstruct/"
savedir = 'images/Postprocessing'

try:
    os.mkdir(savedir)
except:
    pass

nmse_list = []
psnr_list = []

for filename in os.listdir(datadirectory):
    if filename.endswith(".png"):

        ph = np.asarray(Image.open(datadirectory + filename).convert('L')) / 255.
        folder = savedir + '/' + filename[:-4]
        os.mkdir(folder)
        savepath = folder + '/'

        data = ray_trafo(ph)
        fbp = FBP(data)

        xpost = np.asarray(fbp-post_processing_network.predict(co.check_dim(fbp)).reshape((img_height, img_width)))

        nmse = co.NMSE(ph, co.project(xpost))

        psnr_list.append(co.PSNR(ph, co.project(xpost)))
        nmse_list.append(nmse)

        plot(xpost, '{:.2e}'.format(nmse), savepath=savepath, name='Postprocessing')


print("########### PSNR ##############")
print(psnr_list)

print(np.mean(psnr_list))

print("########### NMSE ##############")
print(nmse_list)

print(np.mean(nmse_list))