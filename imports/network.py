from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, DepthwiseConv2D, Input, Concatenate
from keras.models import Model
from keras.regularizers import l2, l1
import keras.initializers as initializers
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.callbacks import Callback
import warnings


class TightFrame:
    def __init__(self, size=512, channels=1):
        self.size = size
        self.channels = channels

    @staticmethod
    def filters(ftype='LL', size=2, ic=1, oc=1, upsampling=False):
        d = {'L': np.array([1. for j in range(size)]), 'H': np.array([(-1.) ** (j + 1) for j in range(size)])}

        weights = np.outer(d[ftype[0]], d[ftype[1]]) / 2.
        out = np.zeros((size, size, ic, oc))

        for i in range(ic):

            if upsampling:
                out[:, :, i, i] = weights

            else:
                for j in range(oc):
                    out[:, :, i, j] = weights

        return initializers.Constant(value=out)

    @staticmethod
    def SequentialLayer(ch, beta=0., ind='0'):
        conv = Conv2D(ch, (3, 3), padding='same', kernel_regularizer=l2(beta), name='SeqConvDown' + ind)
        bn = BatchNormalization(name='SeqBNDown' + ind)
        activation = Activation('relu', name='SeqActDown' + ind)
        layers = [conv, bn, activation]

        def layer(inp):
            out = inp
            for lay in layers:
                out = lay(out)
            return out

        return layer

    def WaveletDecomposition(self, ch, ind=0, alpha=0.):
        wave = []
        for typ in ['HH', 'HL', 'LH', 'LL']:
            fil = self.filters(typ, 2, ch, 1)
            name = ''.join(['WaveletDecomp', typ, str(ind)])

            if typ in ['HH', 'HL', 'LH']:
                lay = DepthwiseConv2D(kernel_size=(2, 2), strides=(2, 2), depthwise_initializer=fil,
                                      name=name, trainable=False, use_bias=False, depth_multiplier=1,
                                      activity_regularizer=l1(alpha))
            else:
                lay = DepthwiseConv2D(kernel_size=(2, 2), strides=(2, 2), depthwise_initializer=fil,
                                      name=name, trainable=False, use_bias=False, depth_multiplier=1)

            wave.append(lay)

        def decomp(inp):

            out = inp
            output = []

            for lay in wave:
                output.append(lay(out))

            return output

        return decomp

    def WaveletComposition(self, ch, ind=0):
        wave = []
        for typ in ['HH', 'HL', 'LH', 'LL']:
            fil = self.filters(typ, 2, ch, ch, upsampling=True)
            name = ''.join(['WaveletConc', typ, str(ind)])
            lay = Conv2DTranspose(ch, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=fil,
                                  name=name, trainable=False, use_bias=False)
            wave.append(lay)

        def comp(InputLayer):
            upsample = []
            for WaveInp, WaveLay in zip(InputLayer, wave):
                upsample.append(WaveLay(WaveInp))
            return upsample

        return comp

    def get_network(self, alpha=1e-3, beta=1e-3, seq_f=64, f=2):
        # Rescaling is necessary due to the implementation of the regularizers/loss in Keras
        scaling_factor = 1. / (self.size ** 2 * self.channels)
        alpha *= scaling_factor
        beta *= scaling_factor

        inp = Input(shape=(self.size, self.size, self.channels))
        output = []

        # DOWNSAMPLING
        seq11 = self.SequentialLayer(seq_f, ind='1_1', beta=beta)(inp)
        seq12 = self.SequentialLayer(f, ind='1_2', beta=beta)(seq11)

        downsampling1 = self.WaveletDecomposition(f, ind=1, alpha=alpha)(seq12)
        output += downsampling1[:3]

        seq21 = self.SequentialLayer(seq_f * 2, ind='2_1', beta=beta)(downsampling1[-1])
        seq22 = self.SequentialLayer(2 * f, ind='2_2', beta=beta)(seq21)

        downsampling2 = self.WaveletDecomposition(2 * f, ind=2, alpha=alpha / 2)(seq22)
        output += downsampling2[:3]

        seq31 = self.SequentialLayer(seq_f * 4, ind='3_1', beta=beta)(downsampling2[-1])
        seq32 = self.SequentialLayer(4 * f, ind='3_2', beta=beta)(seq31)

        downsampling3 = self.WaveletDecomposition(4 * f, ind=3, alpha=alpha / 4)(seq32)
        output += downsampling3[:3]

        seq41 = self.SequentialLayer(seq_f * 8, ind='4_1', beta=beta)(downsampling3[-1])
        seq42 = self.SequentialLayer(8 * f, ind='4_2', beta=beta)(seq41)

        downsampling4 = self.WaveletDecomposition(8 * f, ind=4, alpha=alpha / 8)(seq42)
        output += downsampling4[:3]

        seqLOW1 = self.SequentialLayer(seq_f * 16, ind='LOW_1', beta=beta)(downsampling4[-1])
        seqLOW2 = Conv2D(f * 16, (3, 3), padding='same', kernel_regularizer=l2(beta),
                         activity_regularizer=l1(alpha / 8), activation='relu',
                         name='LowpassOutput')(seqLOW1)

        output += [seqLOW2]

        encoder = Model(inputs=inp, outputs=output)

        decoder_inputs = [Input(shape=[t.value for t in s.shape[1:]]) for s in encoder.outputs]

        # UPSAMPLING
        upsampling4 = self.WaveletComposition(8 * f, ind=4)(decoder_inputs[-4:])
        conc4 = Concatenate()(upsampling4)

        sequp41 = self.SequentialLayer(seq_f * 8, ind='Up4_1', beta=beta)(conc4)
        sequp42 = self.SequentialLayer(seq_f * 8, ind='Up4_2', beta=beta)(sequp41)

        upsampling3 = self.WaveletComposition(4 * f, ind=3)(decoder_inputs[-7:-4] + [sequp42])
        conc3 = Concatenate()(upsampling3)

        sequp31 = self.SequentialLayer(seq_f * 4, ind='Up3_1', beta=beta)(conc3)
        sequp32 = self.SequentialLayer(seq_f * 4, ind='Up3_2', beta=beta)(sequp31)

        upsampling2 = self.WaveletComposition(2 * f, ind=2)(decoder_inputs[-10:-7] + [sequp32])
        conc2 = Concatenate()(upsampling2)

        sequp21 = self.SequentialLayer(seq_f * 2, ind='Up2_1', beta=beta)(conc2)
        sequp22 = self.SequentialLayer(seq_f * 2, ind='Up2_2', beta=beta)(sequp21)

        upsampling1 = self.WaveletComposition(f, ind=1)(decoder_inputs[-13:-10] + [sequp22])
        conc1 = Concatenate()(upsampling1)

        sequp11 = self.SequentialLayer(seq_f, ind='Up1_1', beta=beta)(conc1)
        sequp12 = self.SequentialLayer(seq_f, ind='Up1_2', beta=beta)(sequp11)

        out = Conv2D(1, (1, 1))(sequp12)

        decoder = Model(inputs=decoder_inputs, outputs=out)

        model_out = decoder(encoder(inp))
        model = Model(inputs=inp, outputs=model_out)
        return encoder, decoder, model

    def get_shallow_network(self, alpha=1e-3, beta=1e-3, seq_f=64, f=2):
        scaling_factor = 1. / (self.size ** 2 * self.channels)
        alpha *= scaling_factor
        beta *= scaling_factor

        inp = Input(shape=(self.size, self.size, self.channels))
        seq1 = self.SequentialLayer(seq_f, ind='1_1', beta=beta)(inp)
        seq2 = self.SequentialLayer(f, ind='1_2', beta=beta)(seq1)
        output = self.WaveletDecomposition(f, ind=1, alpha=alpha)(seq2)

        encoder = Model(inputs=inp, outputs=output)
        decoder_inputs = [Input(shape=[t.value for t in s.shape[1:]]) for s in encoder.outputs]

        upsampling = self.WaveletComposition(f, ind=1)(decoder_inputs)
        conc = Concatenate()(upsampling)
        sequp1 = self.SequentialLayer(seq_f, ind='Up4_1', beta=beta)(conc)
        sequp2 = self.SequentialLayer(seq_f, ind='Up4_2', beta=beta)(sequp1)

        out = Conv2D(1, (1, 1))(sequp2)

        decoder = Model(inputs=decoder_inputs, outputs=out)

        model_out = decoder(encoder(inp))
        model = Model(inputs=inp, outputs=model_out)
        return encoder, decoder, model


class AutoencoderCP(Callback):

    def __init__(self, filepath, encoder, decoder, encoder_path, decoder_path, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, mode='auto', period=1):
        super(AutoencoderCP, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.encoder = encoder
        self.decoder = decoder

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                            self.encoder.save(self.encoder_path, overwrite=True)
                            self.decoder.save(self.decoder_path, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                    self.encoder.save(self.encoder_path, overwrite=True)
                    self.decoder.save(self.decoder_path, overwrite=True)

