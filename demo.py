from imports.desyre_optimization import DESYRE
from imports.customobjects import CustomObjects
from keras.models import load_model

import keras.backend as K
import odl
import numpy as np
import matplotlib.pyplot as plt

sess = K.get_session()

# ----------------------------------------
# Set up forward operator using ODL library

size = 512
n_theta = 60
n_s = int(1.5 * size)

reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[size, size], dtype='float32')
angle_partition = odl.uniform_partition(0, np.pi * (1 - 1 / (2 * n_theta)), n_theta)
detector_partition = odl.uniform_partition(-1.5, 1.5, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

Radon = odl.tomo.RayTransform(reco_space, geometry)
FBP = odl.tomo.fbp_op(Radon, filter_type='Hann')

# ----------------------------------------
# Set up DESYRE optimization with loaded networks

co = CustomObjects(sess)

e = load_model('models/encoder.h5', custom_objects=co.custom_objects)
d = load_model('models/decoder.h5', custom_objects=co.custom_objects)

desyre = DESYRE(encoder=e, decoder=d, operator=Radon, size=size, sess=sess)

# Generate a box phantom for demonstration of the algorithm
phantom = np.zeros((size, size))
phantom[(size//2 - 50):(size//2 + 5), (size//2 - 50):(size//2 + 5)] = 1

data = Radon(phantom)
data_noisy = data + np.random.normal(0, 1, data.shape)*np.mean(data)*0.05


x0 = FBP(data_noisy)
x_desyre, err = desyre.fista(x0, data_noisy, niter=10)

plt.semilogy(err)

plt.subplot(131)
plt.imshow(phantom, cmap='bone')

plt.subplot(132)
plt.imshow(x0, cmap='bone')

plt.subplot(133)
plt.imshow(x_desyre, cmap='bone')