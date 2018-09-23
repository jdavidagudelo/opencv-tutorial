import tensorflow as tf
import os

path = '{0}/data/train.csv'.format(os.path.dirname(os.path.abspath(__file__)))
filenames = [path]
defaults = [['0.0']] * 81
dataset = tf.contrib.data.CsvDataset(filenames, defaults, header=True)
print(dataset)
