import tensorflow as tf
import keras.backend as K
from keras.layers import Reshape, Lambda
from keras.models import Model
import odl.contrib.tensorflow
import numpy as np
import odl


class DESYRE:

    def __init__(self, encoder, decoder, operator, size=512, sess=None):
        if sess is None:
            self.sess = K.get_session()
        else:
            self.sess = sess

        self.encoder = encoder
        self.decoder = decoder

        self.operator = operator
        self.size = size

        self.forward = self._net()

        self.input_shape = [tuple([1] + [int(z) for z in s.shape[1:]]) for s in self.decoder.inputs]

        self.xi = [tf.placeholder(tf.float32, shape=s) for s in self.input_shape]
        self.xi_var = [tf.Variable(xi, dtype=tf.float32) for xi in self.xi]

        self.x_init = tf.placeholder(tf.float32, shape=(1, self.size, self.size, 1))
        self.xi_init = self.encoder(self.x_init)

        self.data = tf.placeholder(tf.float32, shape=(1,) + operator.range.shape + (1,))
        self.alpha = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self.weights = self.decaying_weights()

        self.soft = [z.assign(self.shrinkage(z, w*self.alpha)) for z, w in zip(self.xi_var, self.weights)]

        self.data_loss = self._data_discrepancy()
        self.loss = self.data_loss + self.alpha*self._regularizer()

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.gradient_step = self.optimizer.minimize(self.data_loss, var_list=self.xi_var)

        # Variables needed for FISTA implementation

        self.fista_y0 = [tf.Variable(x, dtype=tf.float32) for x in self.xi]
        self.fista_y1 = [tf.Variable(x, dtype=tf.float32) for x in self.xi]

        self.fista_y1_update = [y.assign(xi) for y, xi in zip(self.fista_y1, self.xi_var)]
        self.fista_y0_update = [y.assign(xi) for y, xi in zip(self.fista_y0, self.fista_y1)]

        self.t = tf.placeholder(tf.float32, shape=())
        self.t0 = tf.Variable(self.t)
        self.t1 = tf.Variable(self.t)

        self.update_t1 = self.t1.assign((1 + tf.sqrt(1 + self.t0**2))/2)
        self.update_t0 = self.t0.assign(self.t1)

        self.xi_fista_update = [xi.assign(z1 + ((self.t0 - 1)/(self.t1))*(z1 - z0)) for xi, z1, z0 in zip(self.xi_var, self.fista_y1, self.fista_y0)]
        self.fista_init = tf.variables_initializer(self.xi_var + [self.t0, self.t1] + self.fista_y0 + self.fista_y1)

        pass

    @staticmethod
    def shrinkage(xi, gamma):
        return tf.maximum(tf.abs(xi) - gamma, 0)*tf.sign(xi)

    def _fista_initialization(self, x0):
        xi_inp = self.sess.run(self.xi_init, feed_dict={self.x_init: x0[None, ..., None]})
        xi_feed_dict = {}
        for i in range(len(xi_inp)):
            xi_feed_dict[self.xi[i].name] = xi_inp[i]
        xi_feed_dict[self.t] = 1
        self.sess.run(self.fista_init, feed_dict=xi_feed_dict)
        del xi_inp
        del xi_feed_dict
        pass

    def fista(self, x0, data, alpha=10**(-3), learning_rate=10**(-3), niter=100):
        self._fista_initialization(x0)
        fd = {self.data: data[None, ..., None], self.alpha: alpha, self.lr: learning_rate}
        err = [self.sess.run(self.loss, feed_dict=fd)]
        for it in range(niter):
            self.sess.run(self.gradient_step, feed_dict=fd)
            self.sess.run(self.soft, feed_dict=fd)

            self.sess.run(self.fista_y1_update, feed_dict=fd)
            self.sess.run(self.update_t1)
            self.sess.run(self.xi_fista_update, feed_dict=fd)

            self.sess.run(self.fista_y0_update, feed_dict=fd)
            self.sess.run(self.update_t0)
            err.append(self.sess.run(self.loss, feed_dict=fd))

        x_out = self.sess.run(self.decoder(self.xi_var)).reshape((self.size, self.size))
        return x_out, err

    def _net(self):
        inp = self.decoder.inputs
        out = self.decoder(inp)
        out = Reshape((self.size, self.size, 1))(out)
        operatorlayer = odl.contrib.tensorflow.as_tensorflow_layer(self.operator)
        out = Lambda(operatorlayer)(out)
        net = Model(inputs=inp, outputs=out)
        return net

    def _regularizer(self, q=1):
        norms = [tf.norm(self.xi_var[i] * self.weights[i], ord=q) ** q for i in range(len(self.weights))]
        return K.sum(norms)

    def _data_discrepancy(self, p=2):
        return (1. / p) * tf.norm(self.forward(self.xi_var) - self.data, ord=p) ** p

    def decaying_weights(self):
        w = []
        for s in self.decoder.inputs:
            t = s.shape[1:]
            scale = 2 ** (1 + np.log2(s.shape[1].value) - np.log2(self.size))
            w.append(np.ones([1, ] + [z.value for z in t]) * scale)
        return w
