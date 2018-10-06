import cv2
import pandas as pd
import os
import numpy
import pickle


def encode_images(data_dir, data_csv, batches_count, output_dir):
    data = pd.read_csv(data_csv)
    data_split = numpy.split(data, batches_count)

    for i, current_data in enumerate(data_split):
        batch_label = 'train batch {0} of {1}'.format(i, len(data_split))
        labels = current_data.label.tolist()
        filenames = current_data.filename.tolist()
        data = []
        for filename in filenames:
            img_path = os.path.join(data_dir, filename)
            current_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            current_data = current_data.reshape(current_data.shape[0] * current_data.shape[1])
            data.append(current_data)
        output_file_name = 'train_batch_{0}'.format(i)
        batch_path = os.path.join(output_dir, output_file_name)
        batch_data = {'batch_label': batch_label, 'labels': labels, 'filenames': filenames, 'data': numpy.array(data)}
        with open(batch_path, "wb") as f:
            pickle.dump(batch_data, f)
