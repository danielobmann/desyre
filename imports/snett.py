import tensorflow as tf
import keras.backend as K
from keras.layers import Reshape, Lambda
from keras.models import Model
import odl.contrib.tensorflow
import numpy as np
import odl
from tensorflow.python.ops import math_ops

from sys import getsizeof


class SNETT:

    def __init__(self, encoder, decoder, operator, img_height, img_width, sess=None):
        self.encoder = encoder
        self.decoder = decoder
        self.operator = operator
        self.img_height = img_height
        self.img_width = img_width
        self.forward = self.net()
        if sess is None:
            self.sess = K.get_session()
        else:
            self.sess = sess

        temp = self.encoder.predict(self.check_dim(np.zeros((self.img_height, self.img_width))))

        self.xi = [tf.Variable(tf.convert_to_tensor(temp[i])) for i in range(len(temp))]
        self.sess.run(tf.variables_initializer(self.xi))

        del temp
        print("Xi size", getsizeof(self.xi))

        self.graph = self.sess.graph
        #self.graph.finalize()
        pass

    def net(self):
        inp = self.decoder.inputs
        out = self.decoder(inp)
        out = Reshape((self.img_height, self.img_width, 1))(out)
        operatorlayer = odl.contrib.tensorflow.as_tensorflow_layer(self.operator)
        out = Lambda(operatorlayer)(out)
        net = Model(inputs=inp, outputs=out)
        return net

    @staticmethod
    def regularizer(w=None, q=1):
        if w is None:
            def reg(xi):
                return 0
        else:
            def reg(xi):
                norms = [tf.norm(xi[i] * w[i], ord=q) ** q for i in range(len(w))]
                return K.sum(norms)
        return reg

    @staticmethod
    def total_variation(images):
        pixel_dif1 = images[:, 1:, :-1, :] - images[:, :-1, :-1, :]
        pixel_dif2 = images[:, :-1, 1:, :] - images[:, :-1, :-1, :]
        sq1 = K.square(pixel_dif1)
        sq2 = K.square(pixel_dif2)
        total = K.sqrt(sq1 + sq2)
        tot_var = math_ops.reduce_sum(total, axis=[1, 2, 3])
        return tot_var

    def data_discrepancy(self, data, p=2):

        def dd(xi):
            y = tf.reshape(data, (1,) + data.shape + (1,))
            return (1. / p) * tf.norm(self.forward(xi) - y, ord=p) ** p

        return dd

    def decaying_weights(self):
        w = []
        for s in self.decoder.inputs:
            t = s.shape[1:]
            scale = 2 ** (1 + np.log2(s.shape[1].value) - np.log2(self.img_height))
            w.append(np.ones([1, ] + [z.value for z in t]) * scale)
        return w

    def check_dim(self, x0):
        if len(np.asarray(x0).shape) != 4:
            x0 = np.asarray(x0).reshape((1, self.img_height, self.img_width, 1))
            return x0
        else:
            return x0
        pass

    def get_initial(self, x0):

        temp = self.encoder.predict(self.check_dim(x0))
        [self.xi[i].load(temp[i], session=self.sess) for i in range(len(temp))]
        del temp

        pass

    def GradDes(self, x0, loss, niter=100, lr=10**(-3), momentum=0.80, output_xi=False):
        # Gradient descent for the SNETT reconstruction



        self.get_initial(x0)
        cost = loss(self.xi)

        print("Session size", getsizeof(self.sess))
        print("Graph size", getsizeof(self.graph))

        #opt = tf.train.MomentumOptimizer(lr, momentum=momentum)
        opt = tf.train.GradientDescentOptimizer(lr)
        optimizer = opt.minimize(cost, var_list=self.xi)
        print("Opt size", getsizeof(opt))
        print("Optimizer size", getsizeof(optimizer))
        #self.sess.run(tf.variables_initializer(opt.variables()))


        err = [self.sess.run(cost)]

        for it in range(niter):
            _, c = self.sess.run([optimizer, cost])
            err.append(c)

        out = [self.decoder.predict(self.sess.run(self.xi)).reshape((self.img_height, self.img_width)), err]

        if output_xi:
            out.append(self.sess.run(self.xi))

        return out

    def SoftThreshold(self, x, gamma=1.0):
        return K.sign(x) * K.maximum(0., K.abs(x) - gamma)

    def ISTA(self, x0, loss, w, niter=100, lr=10 ** (-4), output_error=False, alpha=10 ** (-3),
             output_xi=False):
        # ISTA for the SNETT reconstruction

        xi = self.get_initial(x0)
        cost = loss(xi)
        reg = self.regularizer(w=w)(xi)
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost, var_list=xi)
        prox = [xi[i].assign(self.SoftThreshold(xi[i], lr * w[i])) for i in range(len(xi))]

        if output_error:
            err = [self.sess.run(cost + alpha * reg)]

        for it in range(niter):

            self.sess.run([optimizer])
            self.sess.run([prox])

            if output_error:
                err.append([self.sess.run(cost + alpha * reg)][0])
        out = [self.decoder.predict(self.sess.run(xi)).reshape((self.img_height, self.img_width))]

        if output_error:
            out.append(err)

        if output_xi:
            out.append(self.sess.run(xi))

        return out