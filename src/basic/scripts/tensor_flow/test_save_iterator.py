import tensorflow as tf
import os

dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
# Create saveable object from iterator.
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
path = '{0}/out/test.tf'.format(os.path.dirname(os.path.abspath(__file__)))

# Save the iterator state by adding it to the saveable objects collection.
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tf.train.Saver()

with tf.Session() as s1:
    saver.save(s1, path)

# Restore the iterator state.
with tf.Session() as s2:
    saver.restore(s2, path)
