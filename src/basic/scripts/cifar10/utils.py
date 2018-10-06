import tensorflow as tf
import os
import sys
import pickle


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names(train_files=None, validation_files=None, eval_files=None):
    """Returns the file names expected to exist in the input_dir."""
    train_files = ['data_batch_%d' % i for i in range(1, 5)] if train_files is None else train_files
    validation_files = ['data_batch_5'] if validation_files is None else validation_files
    eval_files = ['test_batch'] if eval_files is None else eval_files
    file_names = {'train': train_files, 'validation': validation_files, 'eval': eval_files}
    return file_names


def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict


def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict['data']
            labels = data_dict['labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': bytes_feature(data[i].tobytes()),
                        'label': int64_feature(labels[i])
                    }))
                record_writer.write(example.SerializeToString())


def store_tfrecords(input_dir, batches_folder, output_dir, train_files=None, validation_files=None, eval_files=None):
    file_names = _get_file_names(train_files, validation_files, eval_files)
    input_dir = os.path.join(input_dir, batches_folder)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(output_dir, mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)
    print('Done!')