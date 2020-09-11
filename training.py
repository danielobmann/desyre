import os

from imports.network import TightFrame, AutoencoderCP

from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
import argparse

# -------------------
# Hyperparameter setup

parser = argparse.ArgumentParser()

parser.add_argument("-e", "--epochs", default=150)
parser.add_argument("-b", "--batch", default=6)
parser.add_argument("-a", "--alpha", default=1e-2)
parser.add_argument("-r", "--beta", default=1e-4)
parser.add_argument("-s", "--savepath", default="new")
parser.add_argument("-d", "--datapath", default="data/")

args = vars(parser.parse_args())

batch_size = int(args['batch'])
epochs = int(args['epochs'])

alpha = float(args['alpha'])
beta = float(args['beta'])

size, channels = 512, 1
spe = 800//batch_size
spe_val = 400//batch_size

datapath = args['datapath']
savepath = args['savepath']

model_save = "models/" + savepath + "/"
img_save = "images/" + savepath + "/"

# -------------------
# Set and start up training

if __name__ == '__main__':
    if not os.path.exists("models/"):
        os.mkdir("models/")

    if not os.path.exists("images/"):
        os.mkdir("images/")

    if not os.path.exists(model_save):
        os.mkdir(model_save)

    if not os.path.exists(img_save):
        os.mkdir(img_save)

    tightframe = TightFrame(size=size, channels=channels)
    encoder, decoder, model = tightframe.get_network(alpha=alpha, beta=beta)
    optim = optimizers.Adam(lr=10 ** (-3))

    model.compile(optimizer=optim, loss='mse', metrics=['mse', tightframe.psnr, tightframe.nmse])

    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_directory(datapath, target_size=(size, size),
                                                  batch_size=batch_size, class_mode='input',
                                                  classes=['train'], color_mode='grayscale')

    val_generator = datagen.flow_from_directory(datapath, target_size=(size, size),
                                                batch_size=batch_size, class_mode='input',
                                                classes=['val'], color_mode='grayscale')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                  min_lr=10 ** (-6), cooldown=10, verbose=1, mode='min')

    CP = AutoencoderCP(model_save + 'model.h5', encoder, decoder, model_save + 'encoder.h5', model_save + 'decoder.h5',
                       monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=spe,
                                  validation_data=val_generator, validation_steps=spe_val, callbacks=[reduce_lr, CP])

    plt.semilogy(history.history['loss'], label='Trainloss')
    plt.semilogy(history.history['val_loss'], label='Validationloss')
    plt.legend()

    plt.savefig(img_save + 'train_loss.pdf')
    plt.clf()

    X = val_generator.next()[0]
    xpred = model.predict_on_batch(X)

    for i in range(batch_size):
        plt.subplot(121)
        plt.imshow(X[i, ..., 0], cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.title('Input')

        plt.subplot(122)
        plt.imshow(xpred[i, ..., 0], cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.title('Output')

        plt.savefig(img_save + 'val_image' + str(i) + '.pdf')
        plt.clf()
