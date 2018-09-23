import tensorflow as tf

with tf.Session() as sess:
    inc_dataset = tf.data.Dataset.range(100)
    dec_dataset = tf.data.Dataset.range(0, -100, -1)
    dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
    batched_dataset = dataset.batch(4)

    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
    print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
    print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
