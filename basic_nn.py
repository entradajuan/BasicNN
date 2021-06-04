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

print(train_data)
print(train_y)


my_feature_columns = []
for key in train_data.keys():
  print(key)
  my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)
print(type(my_feature_columns[0]))

model = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[300, 1200, 1200, 100], n_classes=3)

def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    print(type(dataset))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

model.train(input_fn=lambda: input_fn(train_data, train_y, training=True), steps=45000)

eval_result = model.evaluate(input_fn=lambda: input_fn(test_data, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

print(test_data.shape)