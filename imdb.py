import tensorflow as tf

max_len = 200
n_words = 10000
dim_embedding = 512

def load_data():
  (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.imdb.load_data(num_words = n_words)
  X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, max_len)
  X_test =  tf.keras.preprocessing.sequence.pad_sequences(X_test, max_len)
  return (X_train, Y_train), (X_test, Y_test)


(X_train, Y_train), (X_test, Y_test) = load_data()  

print(type(X_train))
print(X_train.shape)
print(X_train[0])
print(Y_train[0])

def create_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Embedding(n_words, dim_embedding, input_length=max_len))
  model.add(tf.keras.layers.Dropout(0.3))
  model.add(tf.keras.layers.GlobalMaxPooling1D())
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  return model


model=create_model()
model.summary()
