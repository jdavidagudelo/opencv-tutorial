from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import pandas
import numpy
import os

import tensorflow as tf

all_data_columns = ['Id',
                    'MSSubClass',
                    'LotArea',
                    'Street',
                    'LotShape',
                    'LandContour',
                    'LotConfig',
                    'LandSlope',
                    'Neighborhood',
                    'Condition1',
                    'Condition2',
                    'BldgType',
                    'HouseStyle',
                    'OverallQual',
                    'OverallCond',
                    'YearBuilt',
                    'YearRemodAdd',
                    'RoofStyle',
                    'RoofMatl',
                    'ExterQual',
                    'ExterCond',
                    'Foundation',
                    'Heating',
                    'HeatingQC',
                    'CentralAir',
                    '1stFlrSF',
                    '2ndFlrSF',
                    'LowQualFinSF',
                    'GrLivArea',
                    'FullBath',
                    'HalfBath',
                    'BedroomAbvGr',
                    'KitchenAbvGr',
                    'TotRmsAbvGrd',
                    'Fireplaces',
                    'PavedDrive',
                    'WoodDeckSF',
                    'OpenPorchSF',
                    'EnclosedPorch',
                    '3SsnPorch',
                    'ScreenPorch',
                    'PoolArea',
                    'MiscVal',
                    'MoSold',
                    'YrSold',
                    'SaleCondition', 'SalePrice']


def normalize_column(data, column):
    column_data = data[column]
    normalized_data = preprocessing.normalize(numpy.array(column_data).reshape(1, -1))
    data[column] = normalized_data[0]


def get_complete_columns(test_data, train_data):
    pass


def read_csv_data_to_pandas(file_name, label):
    data = pandas.read_csv(file_name)


def get_non_empty_columns(data):
    return [column for column in data.columns if not data[column].isna().values.any()]


def get_numeric_columns(data):
    return [column for column in data.columns if getattr(data[column], 'dtype') in [
            numpy.dtype('float64'), numpy.dtype('int64')]]


def read_csv_data(file_name, label=None):
    data = pandas.read_csv(file_name)
    real_columns = all_data_columns
    if all_data_columns is not None:
        if label is None:
            data = data[all_data_columns[:-1]]
            real_columns = all_data_columns[:-1]
        else:
            data = data[all_data_columns]
    if label is not None:
        good_columns = get_columns_with_good_correlation(data, label, minimum_covariance=0.3)
        real_columns = [column for column in all_data_columns if column in good_columns]
    numeric_columns = [
        column for column in real_columns if getattr(data[column], 'dtype') in [
            numpy.dtype('float64'), numpy.dtype('int64')]]
    numeric_columns = [column for column in numeric_columns if column not in [label]]
    for column in numeric_columns:
        normalize_column(data, column)
    feature_columns = []
    current_columns = [column for column in real_columns if column not in [label]]
    for column in current_columns:
        if column in numeric_columns:
            feature_columns.append(tf.feature_column.numeric_column(column))
        else:
            feature_columns.append(
                tf.feature_column.indicator_column(
                    tf.feature_column.categorical_column_with_hash_bucket(
                        column, len(data[column].unique()))))
    if label is not None:
        label_data = data[label]
    else:
        label_data = None
    return data[[column for column in current_columns]], label_data, feature_columns


def convert_data_frame_to_dictionary_numpy(data_frame):
    result = {}
    for column in data_frame.columns:
        result[column] = data_frame[column].values
    return result


def get_columns_with_good_correlation(data_frame, label, minimum_covariance=0.5):
    correlation = data_frame.corr()
    data = correlation[label]
    result = data[lambda x: abs(x) >= minimum_covariance].index.values
    return result


def eval_test_data():
    train_path = '{0}/data/train.csv'.format(os.path.dirname(os.path.abspath(__file__)))
    test_path = '{0}/data/test.csv'.format(os.path.dirname(os.path.abspath(__file__)))
    x, y, feature_columns = read_csv_data(train_path, label='SalePrice')
    x_test, _, _ = read_csv_data(test_path, label=None)
    regressor = tf.estimator.DNNRegressor(
        feature_columns=feature_columns, hidden_units=[10, 10], model_dir='models/housing')
    x_transformed = convert_data_frame_to_dictionary_numpy(x_test)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_transformed, y=None, num_epochs=1, shuffle=False)
    predictions = regressor.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['predictions'] for p in predictions))

    print(y_predicted)


def main(unused_argv):
    # Load dataset
    path = '{0}/data/train.csv'.format(os.path.dirname(os.path.abspath(__file__)))
    x, y, feature_columns = read_csv_data(path, label='SalePrice')

    # Split dataset into train / test
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2)

    x_train = convert_data_frame_to_dictionary_numpy(x_train)
    y_train = y_train.values
    regressor = tf.estimator.DNNRegressor(
        feature_columns=feature_columns, hidden_units=[10, 10], model_dir='models/housing')

    # Train.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)
    regressor.train(input_fn=train_input_fn, steps=2000)

    # Predict.
    x_transformed = convert_data_frame_to_dictionary_numpy(x_test)
    y_test = y_test.values
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_transformed, y=y_test, num_epochs=1, shuffle=False)
    predictions = regressor.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['predictions'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)
    # Score with sklearn.
    score_sklearn = metrics.mean_squared_error(y_predicted, y_test)
    print('MSE (sklearn): {0:f}'.format(score_sklearn))

    # Score with tensorflow.
    scores = regressor.evaluate(input_fn=test_input_fn)
    print('MSE (tensorflow): {0:f}'.format(scores['average_loss']))


if __name__ == '__main__':
    tf.app.run()
