import tensorflow as tf
import keras.backend as K
from keras.layers import Reshape, Lambda
from keras.models import Model
import odl.contrib.tensorflow
import numpy as np
import odl
from tensorflow.python.ops import math_ops


class SNETT:

    def __init__(self, encoder, decoder, operator, img_height, img_width, sess=None):
        if sess is None:
            self.sess = K.get_session()
        else:
            self.sess = sess

        self.graph = tf.get_default_graph()
        self.encoder = encoder
        self.decoder = decoder

        self.operator = operator

        self.img_height = img_height
        self.img_width = img_width

        self.forward = self.net()

        self.input_shape = [tuple([1] + [int(z) for z in s.shape[1:]]) for s in self.decoder.inputs]

        self.xi = [tf.placeholder(tf.float32, shape=s) for s in self.input_shape]
        self.xi_var = [tf.Variable(x, dtype=tf.float32) for x in self.xi]

        self.x_init = tf.placeholder(tf.float32, shape=(1, self.img_height, self.img_width, 1))
        self.xi_init = self.encoder(self.x_init)

        self.data = tf.placeholder(tf.float32)
        self.alpha = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        self.var_init = tf.variables_initializer(self.xi_var)
        self.weights = self.decaying_weights()
        self.shrinkage_weights = [w*self.alpha for w in self.weights]

        self.soft = [z.assign(self.shrinkage(z, w)) for z, w in zip(self.xi_var, self.shrinkage_weights)]

        self.data_loss = self.data_discrepancy(self.xi_var, self.data)
        self.loss = self.data_discrepancy(self.xi_var, self.data)+self.alpha*self.regularizer(self.xi_var, self.weights)

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.optim = self.optimizer.minimize(self.loss, var_list=[self.xi_var])

        self.ista_optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.ista_optim = self.ista_optimizer.minimize(self.data_loss, var_list=[self.xi_var])

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

        #self.graph.finalize()

        pass

    @staticmethod
    def shrinkage(xi, gamma):
        return tf.maximum(tf.abs(xi) - gamma, 0)*tf.sign(xi)

    def var_initialization(self, x0):
        xi_inp = self.sess.run(self.xi_init, feed_dict={self.x_init: self.check_dim(x0)})
        xi_feed_dict = {}
        for i in range(len(xi_inp)):
            xi_feed_dict[self.xi[i].name] = xi_inp[i]

        self.sess.run(self.var_init, feed_dict=xi_feed_dict)
        del xi_inp
        del xi_feed_dict
        pass

    def fista_initialization(self, x0):
        xi_inp = self.sess.run(self.xi_init, feed_dict={self.x_init: self.check_dim(x0)})
        xi_feed_dict = {}
        for i in range(len(xi_inp)):
            xi_feed_dict[self.xi[i].name] = xi_inp[i]
        xi_feed_dict[self.t] = 1
        self.sess.run(self.fista_init, feed_dict=xi_feed_dict)
        del xi_inp
        del xi_feed_dict
        pass

    def ista_optimize(self, x0, data, alpha=10**(-3), learning_rate=10**(-3), niter=100):

        self.var_initialization(x0)
        fd = {self.data: data, self.alpha: alpha, self.lr: learning_rate}
        err = [self.sess.run(self.loss, feed_dict=fd)]


        for it in range(niter):
            self.sess.run([self.ista_optim], feed_dict=fd)
            self.sess.run(self.soft, feed_dict=fd)
            err.append(self.sess.run(self.loss, feed_dict=fd))

        xout = self.sess.run(self.decoder(self.xi_var))
        xout = xout.reshape((self.img_height, self.img_width))
        return xout, err

    def fista_optimize(self, x0, data, alpha=10**(-3), learning_rate=10**(-3), niter=100, xtrue=None):
        self.fista_initialization(x0)
        fd = {self.data: data, self.alpha: alpha, self.lr: learning_rate}
        err = [self.sess.run(self.loss, feed_dict=fd)]
        rel_err = []
        for it in range(niter):
            self.sess.run(self.ista_optim, feed_dict=fd)
            self.sess.run(self.soft, feed_dict=fd)

            self.sess.run(self.fista_y1_update, feed_dict=fd)
            self.sess.run(self.update_t1)
            self.sess.run(self.xi_fista_update, feed_dict=fd)

            self.sess.run(self.fista_y0_update, feed_dict=fd)
            self.sess.run(self.update_t0)
            err.append(self.sess.run(self.loss, feed_dict=fd))
            if not xtrue is None:
                z = self.sess.run(self.decoder(self.xi_var))
                z = z.reshape((self.img_height, self.img_width))
                rel_err.append(np.linalg.norm(z - xtrue)/np.linalg.norm(xtrue))

        xout = self.sess.run(self.decoder(self.xi_var))
        xout = xout.reshape((self.img_height, self.img_width))
        return xout, err, rel_err

    def optimize(self, x0, data, alpha=10**(-3), learning_rate=10**(-3), niter=100):

        self.var_initialization(x0)
        fd = {self.data: data, self.alpha: alpha, self.lr: learning_rate}
        err = [self.sess.run(self.loss, feed_dict=fd)]

        for it in range(niter):
            self.sess.run(self.optim, feed_dict=fd)
            err.append(self.sess.run(self.loss, feed_dict=fd))

        xout = self.sess.run(self.decoder(self.xi_var))
        xout = xout.reshape((self.img_height, self.img_width))

        return xout, err

    def net(self):
        inp = self.decoder.inputs
        out = self.decoder(inp)
        out = Reshape((self.img_height, self.img_width, 1))(out)
        operatorlayer = odl.contrib.tensorflow.as_tensorflow_layer(self.operator)
        out = Lambda(operatorlayer)(out)
        net = Model(inputs=inp, outputs=out)
        return net

    @staticmethod
    def regularizer(xi_inp, w, q=1):
        norms = [tf.norm(xi_inp[i] * w[i], ord=q) ** q for i in range(len(w))]
        return K.sum(norms)

    def data_discrepancy(self, xi_inp, data_inp, p=2):
        y = tf.reshape(data_inp, (1,) + self.operator.range.shape + (1,))
        return (1. / p) * tf.norm(self.forward(xi_inp) - y, ord=p) ** p

    def decaying_weights(self):
        w = []
        for s in self.decoder.inputs:
            t = s.shape[1:]
            scale = 2 ** (1 + np.log2(s.shape[1].value) - np.log2(self.img_height))
            w.append(np.ones([1, ] + [z.value for z in t]) * scale)
        return w

    def decaying_weights_new(self):
        w = []
        for i in range(len(self.decoder.inputs)):
            s = self.decoder.inputs[i]
            t = s.shape[1:]
            scale = 2 ** (1 + np.log2(s.shape[1].value) - np.log2(self.img_height))
            m = np.ones([1, ] + [z.value for z in t])
            if i == (len(self.decoder.inputs) - 1):
                w.append(m*(1/16))
            else:
                w.append(scale*m)
        return w

    @staticmethod
    def check_dim(x0):
        if len(np.asarray(x0).shape) != 4:
            return np.asarray(x0).reshape((1, x0.shape[0], x0.shape[1], 1))
        else:
            return x0
        pass
