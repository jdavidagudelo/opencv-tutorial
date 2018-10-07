import cv2
import pandas as pd
import os
import numpy
import pickle
from sklearn.model_selection import train_test_split


def generate_train_data(data_dir, data_csv, output_dir, limit=None, test_size=0.2):
    """
    Stores pickled dictionary from folder with images.
    This method splits the csv data in a train and test data sets.
    :param data_dir: folder with images.
    :param data_csv: csv file with columns filename and label. The column filename corresponds to
    the name of the file in the data_dir folder and the label column corresponds to the label
    associated to the corresponding file.
    :param output_dir: the folder in wich the data will be stored.
    :param limit: max number of values to process.
    :param test_size: the percentage of the data in the csv document to use for test data.
    :return:
    """
    data = pd.read_csv(data_csv)
    if limit is not None:
        data = data[:limit]
    x_train, x_test, y_train, y_test = train_test_split(data.filename, data.label, test_size=test_size)
    save_data(data_dir, output_dir, x_train, y_train, 'train')
    save_data(data_dir, output_dir, x_test, y_test, 'test')


def save_data(data_dir, output_dir, x_data, y_data, data_name):
    output_file_name = '{0}_batch'.format(data_name)
    data = []
    labels = []
    filenames = []
    for filename, label in zip(x_data, y_data):
        img_path = os.path.join(data_dir, filename)
        current_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        current_data = current_data.reshape(current_data.shape[0] * current_data.shape[1])
        data.append(current_data)
        labels.append(label)
        filenames.append(filename)
    batch_label = '{0} batch 1'.format(data_name)
    batch_data = {'batch_label': batch_label, 'filenames': filenames, 'data': numpy.array(data),
                  'labels': labels}
    batch_path = os.path.join(output_dir, output_file_name)
    with open(batch_path, "wb") as f:
        pickle.dump(batch_data, f)


def encode_images(data_dir, data_csv, batches_count, output_dir, limit=None, data_name='train'):
    data = pd.read_csv(data_csv)
    if limit is not None:
        data = data[:limit]
    data_split = numpy.split(data, batches_count)

    for i, current_data in enumerate(data_split):
        batch_label = '{0} batch {1} of {2}'.format(data_name, i, len(data_split))
        labels = None
        if hasattr(current_data, 'label'):
            labels = current_data.label.tolist()
        filenames = current_data.filename.tolist()
        data = []
        for filename in filenames:
            img_path = os.path.join(data_dir, filename)
            current_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            current_data = current_data.reshape(current_data.shape[0] * current_data.shape[1])
            data.append(current_data)
        output_file_name = '{0}_batch_{1}'.format(data_name, i)
        batch_path = os.path.join(output_dir, output_file_name)
        batch_data = {'batch_label': batch_label, 'filenames': filenames, 'data': numpy.array(data)}
        if labels is not None:
            batch_data['labels'] = labels
        with open(batch_path, "wb") as f:
            pickle.dump(batch_data, f)
