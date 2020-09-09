import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import *


class CustomObjects:
    def __init__(self, sess=None):
        if sess is None:
            self.sess = K.get_session()
        else:
            self.sess = sess
        self.custom_objects = {'bound': self.bound, 'psnr': self.KerasPSNR, 'nmse': self.KerasNMSE}
        pass

    @staticmethod
    def project(x):
        return np.clip(x, 0, 1)

    @staticmethod
    def bound(x):
        return K.minimum(x, K.ones_like(x))

    @staticmethod
    def KerasPSNR(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    @staticmethod
    def KerasNMSE(y_true, y_pred):
        m = K.mean(K.square(y_true - y_pred))
        n = K.mean(K.square(y_true))
        return m/n

    @staticmethod
    def PSNR(x, xhat, maxvalue=1.):
        return 10 * np.log10(maxvalue / np.mean((x - xhat) ** 2))

    @staticmethod
    def NMSE(x, x_hat):
        error = np.mean((x - x_hat)**2)
        normalizer = np.mean(x**2)
        return error/normalizer

    @staticmethod
    def check_dim(x0):
        if len(np.asarray(x0).shape) != 4:
            return np.asarray(x0).reshape((1, x0.shape[0], x0.shape[1], 1))
        else:
            return x0
        pass

    @staticmethod
    def plot(x, text=[], colorbar=True, axis=False, cmap='bone', col='orange', save=None, show=False, title=None):
        fig, ax = plt.subplots()
        im = ax.imshow(x, cmap=cmap, vmin=0.0, vmax=1.0)
        
        if colorbar:
            fig.colorbar(im)
            
        if not axis:
            ax.axis('off')
            
        xstart = 0.01
        ystart = 1-0.05
        ystep = 0.05
        
        for l in range(len(text)):
            t = text[l]
            ax.text(xstart, ystart - l*ystep, t, transform=ax.transAxes, color=col)
        
        if not (save is None):
            fig.savefig(save, format='pdf')
            
        if show:
            fig.show()
            
        elif not show:
            fig.clf()
            
        if not(title is None):
            ax.set_title(title)
        
        pass

    @staticmethod
    def _mark_inset(parent_axes, inset_axes, **kwargs):
        # This code is copied from the matplotlib source code and slightly modified.
        # This is done to avoid the 'connection lines'.
        rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

        if 'fill' in kwargs:
            pp = BboxPatch(rect, **kwargs)
        else:
            fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
            pp = BboxPatch(rect, fill=fill, **kwargs)
        parent_axes.add_patch(pp)

        p1 = BboxConnector(inset_axes.bbox, rect, loc1=1, **kwargs)
        p1.set_clip_on(False)

        p2 = BboxConnector(inset_axes.bbox, rect, loc1=1, **kwargs)
        p2.set_clip_on(False)

        return pp, p1, p2

    def zoomed_plot(self, x, xlim, ylim, zoom=2, text=None, textloc=[], fsize=18, cmap='gray'):

        color = 'orange'
        fig, ax = plt.subplots()
        ax.imshow(np.flipud(x), cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
        ax.axis('off')

        axins = zoomed_inset_axes(ax, zoom, loc=4)

        axins.set_xlim(xlim[0], xlim[1])
        axins.set_ylim(ylim[0], ylim[1])

        self._mark_inset(ax, axins, fc='none', ec=color)

        axins.imshow(np.flipud(x), cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
        axins.patch.set_edgecolor(color)
        axins.patch.set_linewidth('3')
        axins.set_xticks([], [])
        axins.set_yticks([], [])
        #axins.axis('off')

        if not (text is None):
            ax.text(textloc[0], textloc[1], text, color=color, fontdict={'size': fsize}, transform=ax.transAxes)
        pass

    @staticmethod
    def error_plot(err, name, path):
        plt.semilogy(err)
        plt.title(name)
        plt.savefig(path + name + '.pdf', format='pdf')
        plt.clf()
        pass
