import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, UpSampling2D, BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import smart_resize
from tensorflow.python.keras.utils.vis_utils import plot_model

save_plots = True
target_size = 100

# download dataset from http://vis-www.cs.umass.edu/lfw/lfw.tgz
images = glob.glob(os.path.join('data', 'lfw', '**', '*.jpg'))

hi_res_dataset = []
lo_res_dataset = []
ratio = 0.40
low_res_dim = (int(target_size * ratio), int(target_size * ratio))

# for i in tqdm(images[:1000]):
for i in tqdm(images):
    hr_img = image.load_img(i, target_size=(target_size, target_size, 3))
    hr_img = image.img_to_array(hr_img)
    lr_img = smart_resize(hr_img, size=low_res_dim)
    lr_img = smart_resize(lr_img, size=(target_size, target_size))
    hr_img = hr_img / 255.
    lr_img = lr_img / 255.
    hi_res_dataset.append(hr_img)
    lo_res_dataset.append(lr_img)

hi_res_dataset = np.array(hi_res_dataset)
lo_res_dataset = np.array(lo_res_dataset)

# Hyperparameters
learning_rate = 0.001
n_epochs = 50
batch_size = 256
validation_split = 0.1
activation = 'relu'

# split data into train and validation data
x_tr_lo, x_vl_lo, y_tr_hi, y_vl_hi = train_test_split(lo_res_dataset, hi_res_dataset, test_size=validation_split)

fig, ax = plt.subplots(3, 2, figsize=(10, 15), dpi=100)
idxs = random.sample(range(x_tr_lo.shape[0]), 3)
for i in range(3):
    ax[i, 0].imshow(x_tr_lo[idxs[i]])
    ax[i, 1].imshow(y_tr_hi[idxs[i]])

plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('results', 'imgAE_train.png'))
else:
    plt.show()

autoencoder = Sequential()
autoencoder.add(Conv2D(input_shape=(target_size, target_size, 3), filters=256, kernel_size=(3, 3),
                       activation=activation, padding='same'))
autoencoder.add(BatchNormalization(axis=-1))
autoencoder.add(Conv2D(filters=128, kernel_size=(3, 3), activation=activation, padding='same'))
autoencoder.add(BatchNormalization(axis=-1))
autoencoder.add(MaxPool2D(pool_size=(2, 2)))
autoencoder.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activation, padding='same'))
autoencoder.add(BatchNormalization(axis=-1))

autoencoder.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activation, padding='same'))
autoencoder.add(BatchNormalization(axis=-1))
autoencoder.add(UpSampling2D(size=(2, 2)))
autoencoder.add(Conv2D(filters=128, kernel_size=(3, 3), activation=activation, padding='same'))
autoencoder.add(BatchNormalization(axis=-1))
autoencoder.add(Conv2D(filters=256, kernel_size=(3, 3), activation=activation, padding='same'))
autoencoder.add(BatchNormalization(axis=-1))
autoencoder.add(Conv2D(filters=3, kernel_size=(3, 3), activation=activation, padding='same'))

# Optimizer (https://keras.io/api/optimizers/)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Loss (https://keras.io/api/losses/)
loss = keras.losses.mean_squared_error

autoencoder.compile(optimizer=optimizer, loss=loss)

autoencoder.summary()
plot_model(autoencoder, to_file=os.path.join('results', 'imgAE.png'), show_shapes=True)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')

history = autoencoder.fit(x_tr_lo, y_tr_hi, epochs=n_epochs, batch_size=batch_size, shuffle=True,
                          validation_data=(x_vl_lo, y_vl_hi),
                          callbacks=[early_stop]
                          )

# Plot training history
_, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.plot(history.history['loss'], 'r')
ax.plot(history.history['val_loss'], 'g')
ax.set_xlabel('Num of Epochs')
ax.set_ylabel('Loss')
ax.set_title('Training Loss vs Validation Loss')
ax.legend(['train', 'validation'])

plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('results', 'imgAE_history.png'))
else:
    plt.show()

fig, ax = plt.subplots(3, 2, figsize=(10, 15), dpi=100)
idxs = random.sample(range(x_vl_lo.shape[0]), 3)
predictions = autoencoder.predict(x_vl_lo[idxs])
for i in range(3):
    ax[i, 0].imshow(x_vl_lo[idxs[i]])
    ax[i, 1].imshow(predictions[i])

plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('results', 'imgTest.png'))
else:
    plt.show()
