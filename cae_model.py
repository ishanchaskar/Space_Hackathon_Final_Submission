import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input, BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import LeakyReLU
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
import rasterio
import os
import cv2
import numpy as np
from tqdm import tqdm
import math

HEIGHT = 100
WIDTH = 100


def load_data(path, test_split=0.1):
    img_data = []
    files = os.listdir(path)
    for i in tqdm(files):
        src = rasterio.open(path + '/' + i)
        img = src.read()

        if img.shape != (3, HEIGHT, WIDTH):
            continue
        img_data.append(img)
    size = math.floor(len(img_data) * test_split)
    # print(len(img_data), len(img_data[0]))
    return np.array(img_data)[:-size], np.array(img_data)[-size:]


X_path = r"D:\Downloads\space_imps\segment_boonidhi"
y_path = r"D:\Downloads\space_imps\segment_bhuvan"

if not os.path.exists("datasets/X_train.npy"):
    X_train, X_test = load_data(X_path)
    y_train, y_test = load_data(y_path)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)

    X_train = X_train / (2 ** 10 - 1)
    X_test = X_test / (2 ** 10 - 1)
    y_train = y_train / (2 ** 10 - 1)
    y_test = y_test / (2 ** 10 - 1)

    # check
    X_train = X_train.reshape(X_train.shape[0], HEIGHT, WIDTH, 3)
    X_test = X_test.reshape(X_test.shape[0], HEIGHT, WIDTH, 3)

    y_train = y_train.reshape(y_train.shape[0], HEIGHT, WIDTH, 3)
    y_test = y_test.reshape(y_test.shape[0], HEIGHT, WIDTH, 3)

    np.save('datasets/X_train.npy', X_train)
    np.save('datasets/X_test.npy', X_test)
    np.save('datasets/y_train.npy', y_train)
    np.save('datasets/y_test.npy', y_test)

else:
    X_train = np.load('datasets/X_train.npy')
    X_test = np.load('datasets/X_test.npy')
    y_train = np.load('datasets/y_train.npy')
    y_test = np.load('datasets/y_test.npy')

input_layer = Input(shape=(100, 100, 3))

# Encoder
x = Conv2D(64, (3, 3), padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPool2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_layer, decoded)
autoencoder.summary()

autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"])

es = EarlyStopping(monitor='val_acc', patience=100)

checkpoint_filepath = 'checkpoints/checkpoint-finaltry-{epoch:02d}-{val_acc:0.4f}.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)


callbacks_list = [model_checkpoint_callback, es]

autoencoder.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.2)

autoencoder.evaluate(x=X_test, y= y_test, batch_size=64, verbose=1)

