import tensorflow as tf
import numpy as np
from tensorflow import keras

NB_CLASSES = 10

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

X_train = X_train/ 255.0
X_test = X_test/ 255.0

print(Y_train[0])

Y_train = keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test =  keras.utils.to_categorical(Y_test, NB_CLASSES)

print(Y_train[0])

N_HIDDEN = 2500
DROPOUT = 0.25

#build the model
model = tf.keras.models.Sequential()

model.add(keras.layers.Dense(N_HIDDEN,input_shape=(RESHAPED,),
   		name='input_layer', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(N_HIDDEN,
   		name='hidden_layer_1', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(N_HIDDEN,
   		name='hidden_layer_2', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(N_HIDDEN,
   		name='hidden_layer_3', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(NB_CLASSES,
   		name='output_layer', activation='softmax'))

model.summary()

model.compile(optimizer='Adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 50
EPOCHS = 7
VERBOSE = True 
VALIDATION_SPLIT = 0.2

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
		verbose=VERBOSE, validation_split=VALIDATION_SPLIT)


test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

# making prediction
predictions = model.predict(X_test)



