import tensorflow as tf


def load_data():
  (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.imdb.load_data()
  return (X_train, Y_train), (X_test, Y_test)

#def create_model():


(X_train, Y_train), (X_test, Y_test) = load_data()  

print(type(X_train))
print(X_train.shape)
print(X_train[0])
print(Y_train[0])


