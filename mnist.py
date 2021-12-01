import tensorflow as tf
import numpy as np
from tensorflow import keras

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(type(X_train))
print(X_train.shape)



