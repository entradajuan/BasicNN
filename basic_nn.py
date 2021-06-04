import numpy as np
import pandas as pd

%tensorflow_version 2.x
import tensorflow as tf
print(tf.version)



train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

print('type: ' , type(train_path))
print('train_path = ', train_path)

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'] 

train_data = pd.read_csv(train_path, names= CSV_COLUMN_NAMES, header = 0)
test_data = pd.read_csv(test_path, names= CSV_COLUMN_NAMES, header = 0)

train_y = train_data.pop('Species')
test_y = test_data.pop('Species')