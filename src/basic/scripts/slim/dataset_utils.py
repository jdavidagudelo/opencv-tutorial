import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import pickle

LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
    """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


def read_images(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save_tf_records(training_output, testing_output, training_filename, testing_filename, width, height, channels):
    with tf.python_io.TFRecordWriter(training_output) as tf_record_writer:
        data = read_images(training_filename)
        images = data.get('data')
        labels = data.get('labels')
        save_images_to_tf_records(tf_record_writer, images, labels, width, height, channels)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_output) as tf_record_writer:
        data = read_images(testing_filename)
        images = data.get('data')
        labels = data.get('labels')
        save_images_to_tf_records(tf_record_writer, images, labels, width, height, channels)


def save_images_to_tf_records(tf_record_writer, images, labels, width, height, channels):
    shape = (width, height, channels)
    num_images = len(images)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)

        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
                sys.stdout.flush()
                current_image = images[j]
                current_image = current_image.reshape(shape)
                png_string = sess.run(encoded_png, feed_dict={image: current_image})

                example = image_to_tfexample(
                    png_string, 'png'.encode(), width, height, labels[j])
                tf_record_writer.write(example.SerializeToString())


def get_split(split_name, num_samples, num_classes, output_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading MNIST.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
    file_pattern = '{0}.tfrecords'.format(split_name)
    file_pattern = os.path.join(output_dir, file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': tf.contrib.slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
        'label': tf.contrib.slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    #if has_labels(dataset_dir):
    #    labels_to_names = read_label_file(dataset_dir)

    return tf.contrib.slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions={
            'image': 'A [28 x 28 x 1] grayscale image.',
            'label': 'A single integer between 0 and 9',
        },
        labels_to_names=labels_to_names)
