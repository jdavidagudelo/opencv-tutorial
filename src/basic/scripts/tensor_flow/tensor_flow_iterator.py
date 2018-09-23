import tensorflow as tf

dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()

for i in range(100):
    value = sess.run(next_element)
    assert i == value
sess.close()