import numpy as np
from tensorflow import keras
import tensorflow as tf
from numpy import asarray
from numpy import save
import numpy as np
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show
import matplotlib.image as mpimg

X_p = np.load('con2d_H_test_x.npy')
y_p = np.load('con2d_H_test_y.npy')

model1 = keras.models.load_model('conv2d_dense001_horizontal.h5')

#-1, 1, 100, 50, 1


for f in range(0, len(X_p)):
    #x, y = p(file)
    x = X_p[f].reshape(50, 100)
    imshow(x)
    show()
    z = model1.predict(y_p[f].reshape(-1, 1, 50, 100, 1))
    imshow(z.reshape(50, 100))
    show()
    c = str(input("merge?"))
    if c =='y':
        z = np.concatenate((x.reshape(50, 100), z.reshape(50, 100)), axis=0)
        imshow(z)
        show()

model1.summary()

