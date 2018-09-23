import tensorflow as tf
import os
from basic.scripts.house_prices import utils

path = '{0}/data/train.csv'.format(os.path.dirname(os.path.abspath(__file__)))
file_names = [path]
defaults = [['0.0']] * 81

record_defaults = [
    tf.float64, tf.float64, tf.string, tf.float64, tf.float64,
    tf.string, tf.string, tf.string,
    tf.string, tf.string, tf.string, tf.string, tf.string,
    tf.string, tf.string, tf.string, tf.string, tf.float64,
    tf.float64, tf.float64, tf.float64,
    tf.string, tf.string, tf.string, tf.string, tf.string, tf.float64,
    tf.string, tf.string, tf.string,
    tf.string, tf.string, tf.string, tf.string, tf.float64,
    tf.string, tf.float64, tf.float64, tf.float64,
    tf.string, tf.string, tf.string, tf.string, tf.float64, tf.float64, tf.float64,
    tf.float64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64,
    tf.float64, tf.string, tf.float64, tf.string, tf.float64,
    tf.string, tf.string, tf.float64, tf.string, tf.float64, tf.float64,
    tf.string, tf.string, tf.string, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64,
    tf.float64, tf.string, tf.string, tf.string, tf.float64, tf.float64, tf.float64,
    tf.string, tf.string, tf.float64
]
columns_names = [
    'Id', 'MSSubClass',
    'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
    'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
    'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
    'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'
]

feature_columns, record_defaults = utils.get_column_features_from_csv_data_set(file_names, defaults, columns_names)

label = 'SalePrice'


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=record_defaults)

    # Pack the result into a dictionary
    features = dict(zip(columns_names, fields))
    # Separate the label from the features
    current_label = features.get(label)
    print(label in features)
    return features, current_label


def csv_input_fn(csv_path, batch_size):
    # Create a data_set containing the text lines.
    data_set = tf.data.TextLineDataset(csv_path).skip(1)
    # Parse each line.
    data_set = data_set.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    data_set = data_set.shuffle(1000).batch(batch_size)

    # Return the data_set.
    return data_set


#data_set = tf.contrib.data.CsvDataset(file_names, defaults, header=True)
#data_set = data_set.map(lambda *x: tf.convert_to_tensor(x))

# Build the estimator
est = tf.estimator.LinearRegressor(feature_columns[:1])
# Train the estimator
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda: csv_input_fn(path, batch_size))
