import tensorflow as tf
import numpy as np
from tensorflow import keras

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(type(X_train))
print(X_train.shape)

image = X_train[0]
print(image.shape)
RESHAPED = image.shape[0] * image.shape[1]

print(RESHAPED)

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], RESHAPED)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], RESHAPED)
print(X_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/ 255
X_test = X_test/ 255

X_train[0]



