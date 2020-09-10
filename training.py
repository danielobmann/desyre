from imports.network import TightFrame, AutoencoderCP

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import matplotlib.pyplot as plt

# -------------------
# Hyperparameter setup
alpha = 1e-2
beta = 1e-4
f = 2
seq_f = 64

size, channels = 512, 1

tightframe = TightFrame(size=size, channels=channels)
encoder, decoder, model = tightframe.get_network(alpha=alpha, beta=beta, seq_f=seq_f, f=f)
optim = optimizers.Adam(lr=10 ** (-3))

model.compile(optimizer=optim, loss='mse', metrics=['mse', tightframe.psnr, tightframe.nmse])

# -------------------
# Set up training

batch_size = 6
epochs = 150
spe = 800//batch_size
spe_val = 400//batch_size

datapath = 'data/'
savepath = 'models/new/'
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(datapath, target_size=(size, size),
                                              batch_size=batch_size, class_mode='input',
                                              classes=['train'], color_mode='grayscale')

val_generator = datagen.flow_from_directory(datapath, target_size=(size, size),
                                            batch_size=batch_size, class_mode='input',
                                            classes=['val'], color_mode='grayscale')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                              min_lr=10 ** (-6), cooldown=10, verbose=1, mode='min')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
                           verbose=1, mode='min')
checkpoint = ModelCheckpoint(savepath + 'model.h5', monitor='val_loss',
                             mode='min', save_best_only=True, verbose=1)

CP = AutoencoderCP(savepath + 'model.h5', encoder, decoder, savepath + 'encoder.h5', savepath + 'decoder.h5',
                   monitor='val_loss', mode='min', save_best_only=True, verbose=1)

history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=spe,
                              validation_data=val_generator, validation_steps=spe_val, callbacks=[reduce_lr, CP])


plt.semilogy(history.history['loss'], label='Trainloss')
plt.semilogy(history.history['val_loss'], label='Validationloss')
plt.legend()

plt.savefig('images/train_loss.pdf')
plt.clf()

X = val_generator.next()[0]
xpred = model.predict_on_batch(X)

for i in range(batch_size):
    plt.subplot(121)
    plt.imshow(X[i, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.title('Input')

    plt.subplot(122)
    plt.imshow(xpred[i, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.title('Output')

    plt.savefig('images/val_image' + str(i) + '.pdf')
    plt.clf()