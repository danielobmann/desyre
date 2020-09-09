from imports.customobjects import *
from imports.snett_efficient import *
from imports.regularization import *

import keras.backend as K
from keras.models import load_model
import numpy as np
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os
import time
import matplotlib

sess = K.get_session()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess = K.get_session()
co = CustomObjects(sess)

nettype = 'tightframe/models/'
nettype_post = 'postprocessing'
e = load_model('networks/' + nettype + 'good/encoder.h5', custom_objects=co.custom_objects)
d = load_model('networks/' + nettype + 'good/decoder.h5', custom_objects=co.custom_objects)
pp = load_model('networks/' + nettype_post + '/models/postprocessing.h5', custom_objects=co.custom_objects)


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

regularizer = regularization_methods(ray_trafo)

file = datadirectory + 'reconstruct/L506_2.png'
ph = np.asarray(Image.open(file).convert('L'))/255.
data = ray_trafo(ph)
data_noisy = data + odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.05 # 5% additive Gaussian noise

fbp = FBP(data)
fbp_noisy = FBP(data_noisy)

xlim = [230, 280] #506_35 -> [220, 270] #506_2 -> [30, 100] #506_20 -> [130, 200]
ylim = [390, 440] #506_35 -> [120, 170] #506_2 -> [280, 330]  #506_20 -> [130, 200]
textloc = [0.01, 0.94]
fsize = 20
zoom = 2.5
cmap = 'gray'

niter_dsr = 2000
niter_tv = 4000
niter_wave = 500

t0 = time.time()
xista, errista, ista_rel_err = snett.fista_optimize(co.project(FBP(data)), data, alpha=10**(-6), learning_rate=10**(-3), niter=niter_dsr, xtrue=ph)
print("DSR time per iteration", (time.time()-t0)/niter_dsr)

t0 = time.time()
xista_noisy, errista_noisy, ista_rel_err_noisy = snett.fista_optimize(co.project(FBP(data_noisy)), data_noisy, alpha=3*10**(-5), learning_rate=10**(-3), niter=niter_dsr, xtrue=ph)
print("DSR time per iteration (noisy)", (time.time()-t0)/niter_dsr)

t0 = time.time()
xwave, errwave, wave_rel_err = regularizer.Wavelet(co.project(FBP(data)), data, niter=niter_wave, alpha=10**(-8), nlevels=8, xtrue=ph)
print("Wavelet time per iteration", (time.time()-t0)/niter_wave)

t0 = time.time()
xwave_noisy, errwave_noisy, wave_rel_err_noisy = regularizer.Wavelet(co.project(FBP(data_noisy)), data_noisy, niter=niter_wave, alpha=2*10**(-7), nlevels=8, xtrue=ph)
print("Wavelet time per iteration (noisy)", (time.time()-t0)/niter_wave)

t0 = time.time()
xtv, errtv, tv_rel_err = regularizer.TV(co.project(FBP(data)), data, niter=niter_tv, alpha=5*10**(-5), xtrue=ph)
print("TV time per iteration", (time.time()-t0)/niter_tv)

t0 = time.time()
xtv_noisy, errtv_noisy, tv_rel_err_noisy = regularizer.TV(co.project(FBP(data_noisy)), data_noisy, niter=niter_tv, alpha=10**(-4), xtrue=ph)
print("TV time per iteration (noisy)", (time.time()-t0)/niter_tv)

xpost = np.asarray(fbp-pp.predict(co.check_dim(fbp)).reshape((img_height, img_width)))
xpost_noisy = np.asarray(fbp_noisy-pp.predict(co.check_dim(fbp_noisy)).reshape((img_height, img_width)))

post_rel_err = [np.linalg.norm(xpost - ph)/np.linalg.norm(ph)]
post_rel_err_noisy = [np.linalg.norm(xpost_noisy - ph)/np.linalg.norm(ph)]

fbp_rel_err = [np.linalg.norm(fbp - ph)/np.linalg.norm(ph)]
fbp_rel_err_noisy = [np.linalg.norm(fbp_noisy - ph)/np.linalg.norm(ph)]

REC = [xista, xwave, xtv, xpost]
NAME = ['DESYRE', 'Wavelet', 'TV', 'Post-processing']

for rec, name in zip(REC, NAME):
    nmse = co.NMSE(ph, co.project(rec))
    co.zoomed_plot(rec, xlim=xlim, ylim=ylim, textloc=textloc, text=name, cmap=cmap, zoom=zoom)
    plt.savefig('images/reconstructions/' + name + '.pdf', format='pdf')
    plt.clf()

REC = [xista_noisy, xwave_noisy, xtv_noisy, xpost_noisy]

for rec, name in zip(REC, NAME):
    nmse = co.NMSE(ph, co.project(rec))
    co.zoomed_plot(rec, xlim=xlim, ylim=ylim, textloc=textloc, text=name, cmap=cmap, zoom=zoom)
    plt.savefig('images/reconstructions/' + name + '_noisy.pdf', format='pdf')
    plt.clf()


#nmse = co.NMSE(ph, FBP(data))
co.zoomed_plot(FBP(data), xlim=xlim, ylim=ylim, textloc=textloc, cmap=cmap, zoom=zoom, text='FBP')
plt.savefig('images/reconstructions/FBP.pdf', format='pdf')
plt.clf()

#nmse = co.NMSE(ph, FBP(data_noisy))
co.zoomed_plot(FBP(data_noisy), xlim=xlim, ylim=ylim, textloc=textloc, cmap=cmap, zoom=zoom, text='FBP')
plt.savefig('images/reconstructions/FBP_noisy.pdf', format='pdf')
plt.clf()

co.zoomed_plot(ph, xlim=xlim, ylim=ylim, textloc=textloc, cmap=cmap, zoom=zoom, text='True')
plt.savefig('images/reconstructions/GROUNDTRUTH.pdf', format='pdf')
plt.clf()

fig, ax = plt.subplots(1, 1)
ax.imshow(data, cmap='gray')
ax.axis('off')
ax.set_aspect(n_s/n_theta)
plt.savefig('images/reconstructions/DATA.pdf', format='pdf')
plt.clf()

fig, ax = plt.subplots(1, 1)
ax.imshow(data_noisy, cmap='gray')
ax.axis('off')
ax.set_aspect(n_s/n_theta)
plt.savefig('images/reconstructions/DATA_noisy.pdf', format='pdf')
plt.clf()


font = {'size': 12}
matplotlib.rc('font', **font)

plt.semilogy(errista, label="No noise")
plt.semilogy(errista_noisy, label="Noisy")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.savefig("images/reconstructions/ERRORPLOT_DSR.pdf", format='pdf')
plt.clf()

plt.semilogy(errwave, label="No noise")
plt.semilogy(errwave_noisy, label="Noisy")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.savefig("images/reconstructions/ERRORPLOT_WAVE.pdf", format='pdf')
plt.clf()

plt.semilogy(errtv, label="No noise")
plt.semilogy(errtv_noisy, label="Noisy")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.savefig("images/reconstructions/ERRORPLOT_TV.pdf", format='pdf')
plt.clf()


plt.semilogy(errista, color='b', label="DSR (noise-free)")
plt.semilogy(errista_noisy, color='b', linestyle='dashed', label="DSR (noisy)")

plt.semilogy(errwave, color='g', label="Wavelet (noise-free)")
plt.semilogy(errwave_noisy, color='g', linestyle='dashed', label="Wavelet (noisy)")

plt.semilogy(errtv, color='r', label="TV (noise-free)")
plt.semilogy(errtv_noisy, color='r', linestyle='dashed', label="TV (noisy)")

plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.savefig("images/reconstructions/ERRORPLOT.pdf", format='pdf')
plt.clf()


# Relative error plot (noise-free data)
plt.semilogx(wave_rel_err, label='Wavelet')
plt.semilogx(tv_rel_err, label='TV')
plt.semilogx(ista_rel_err, label='DSR')
plt.semilogx(niter_tv*post_rel_err, linestyle='dashed', label='Post-processing')
#plt.semilogx(niter_tv*fbp_rel_err, linestyle='dashed', label='FBP')
plt.legend(loc='lower left')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.savefig('images/reconstructions/RELATIVE_ERROR.pdf')
plt.clf()

# Relative error plot (noisy data)
plt.semilogx(wave_rel_err_noisy, label='Wavelet')
plt.semilogx(tv_rel_err_noisy, label='TV')
plt.semilogx(ista_rel_err_noisy, label='DSR')
plt.semilogx(niter_tv*post_rel_err_noisy, linestyle='dashed', label='Post-processing')
#plt.semilogx(niter_tv*fbp_rel_err_noisy, linestyle='dashed', label='FBP')
plt.legend(loc='lower left')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.savefig('images/reconstructions/RELATIVE_ERROR_noisy.pdf')
plt.clf()

