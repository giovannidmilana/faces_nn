from os import listdir
from numpy import asarray
from numpy import save
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# define location of dataset
folder = 'faces/'
X_p, y_p = list(), list()
i = 0
# enumerate files in the directory
for file in listdir(folder):
    # load image
    i += 1
    photo = load_img(folder + file, target_size=(100, 100))
    # convert to numpy array
    #print(file)
    photo = photo.convert('L')
    photo = img_to_array(photo)
    # store
    #photo = photo.convert('L')
    x, y = np.split(photo, 2, axis=1)
    x /= 255.0
    y /= 255.0
    X_p.append(x)
    y_p.append(y)
    print(i)

# convert to a numpy arrays
X_p = asarray(X_p)
y_p = asarray(y_p)

X = X_p[:70000]
y = y_p[:70000]


print(X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=10)

print(X_train.shape)


batch_size = 32          # Batch size (you may try different values)
epochs = 2              # Epoch (you may try different values)


X_train = X_train.reshape((69300, 1, 100, 50, 1))
y_train = y_train.reshape((69300, 1, 100, 50, 1))
X_test = X_test.reshape((700, 1, 100, 50, 1))
y_test = y_test.reshape((700, 1, 100, 50, 1))


model = tf.keras.models.Sequential()

model.add(layers.ConvLSTM2D(1, kernel_size=(3, 3), return_sequences=True, padding="same", data_format='channels_last', input_shape=(1, 100, 50, 1)))
model.add(layers.BatchNormalization())


model.add(layers.Reshape((1, 5000)))
model.add(layers.Dense(5000, activation='relu'))

model.add(layers.Reshape((1, 100, 50, 1), input_shape=(1, 5000)))
model.add(layers.ConvLSTM2D(1, kernel_size=(3, 3), return_sequences=True, padding="same", data_format='channels_last', input_shape=(1, 100, 50, 1)))
model.add(layers.BatchNormalization())

model.add(layers.Reshape((1, 5000)))
model.add(layers.Dense(5000, activation='relu'))

model.add(layers.Reshape((1, 100, 50, 1), input_shape=(1, 5000)))
model.add(layers.ConvLSTM2D(1, kernel_size=(3, 3), return_sequences=True, padding="same", data_format='channels_last', input_shape=(1, 100, 50, 1)))
model.add(layers.BatchNormalization())

model.add(layers.Reshape((1, 5000)))
model.add(layers.Dense(5000, activation='relu'))

model.add(layers.Reshape((1, 100, 50, 1), input_shape=(1, 5000)))
model.add(layers.ConvLSTM2D(1, kernel_size=(3, 3), return_sequences=True, padding="same", data_format='channels_last', input_shape=(1, 100, 50, 1)))

model.compile(loss='mae', optimizer='adam')
model.fit(x=X_train,
          y=y_train,
          epochs=epochs,
          batch_size = batch_size
         )

save('con2d_V_test_x.npy', X_test)
save('con2d_V_test_y.npy', y_test)

model.save('conv2d_dense001_vertical.h5')

