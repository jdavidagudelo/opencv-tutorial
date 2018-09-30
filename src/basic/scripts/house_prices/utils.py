import tensorflow as tf
from sklearn import preprocessing


def is_valid_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def get_features_dictionary_from_csv(file_names, defaults, column_names):
    data_set = tf.contrib.data.CsvDataset(file_names, defaults, header=True)
    data_set = data_set.map(lambda *x: tf.convert_to_tensor(x))
    dictionary = {c: [] for c in column_names}
    value = data_set.make_one_shot_iterator().get_next()
    with tf.Session() as session:
        while True:
            try:
                v = session.run(value)
                for index, z in enumerate(v):
                    dictionary[column_names[index]].append(z)
            except Exception:
                break
    return dictionary


def get_feature_columns_tensorflow(column_types):
    result = []
    for column in column_types:
        column_type = column_types.get(column)
        if column_type == 'numeric':
            result.append(tf.feature_column.numeric_column(column))
        else:
            bucket_size = int(column_type.split(':')[1])
            result.append(tf.feature_column.categorical_column_with_hash_bucket(
                column, bucket_size))
    return result


def get_column_types_from_csv(file_names, defaults, column_names):
    data_set = tf.contrib.data.CsvDataset(file_names, defaults, header=True)
    data_set = data_set.map(lambda *x: tf.convert_to_tensor(x))
    dictionary = {c: [] for c in column_names}
    value = data_set.make_one_shot_iterator().get_next()
    with tf.Session() as session:
        while True:
            try:
                v = session.run(value)
                for index, z in enumerate(v):
                    dictionary[column_names[index]].append(z)
            except Exception:
                break
    invalid_floats = {}
    for column in column_names:
        invalid_floats[column] = [v for v in dictionary.get(column) if not is_valid_float(v)]
    result = {}
    for column in column_names:
        column_data = dictionary.get(column)
        if len(invalid_floats.get(column)) == 0:
            result[column] = 'numeric'
        else:
            column_data = set(column_data)
            result[column] = 'categorical:{0}'.format(len(column_data))
    return result


def get_default_features_columns_from_csv(file_names, defaults, column_names):
    data_set = tf.contrib.data.CsvDataset(file_names, defaults, header=True)
    data_set = data_set.map(lambda *x: tf.convert_to_tensor(x))
    dictionary = {c: [] for c in column_names}
    value = data_set.make_one_shot_iterator().get_next()
    with tf.Session() as session:
        while True:
            try:
                v = session.run(value)
                for index, z in enumerate(v):
                    dictionary[column_names[index]].append(z)
            except Exception:
                break
    invalid_floats = {}
    for column in column_names:
        invalid_floats[column] = [v for v in dictionary.get(column) if not is_valid_float(v)]
    result = []
    record_defaults = []
    for column in column_names:
        column_data = dictionary.get(column)
        if len(invalid_floats.get(column)) == 0:
            result.append(tf.feature_column.numeric_column(column))
            record_defaults.append([0.0])
        else:
            column_data = set(column_data)
            record_defaults.append([''])
            result.append(tf.feature_column.categorical_column_with_hash_bucket(
                        column,
                        len(column_data)))
    return result, record_defaults


def get_column_features_from_csv_data_set(file_names, defaults, column_names):
    data_set = tf.contrib.data.CsvDataset(file_names, defaults, header=True)
    data_set = data_set.map(lambda *x: tf.convert_to_tensor(x))
    features = []
    dictionary = {c: set() for c in column_names}
    value = data_set.make_one_shot_iterator().get_next()
    with tf.Session() as session:
        while True:
            try:
                v = session.run(value)
                index = 0
                for z in v:
                    try:
                        float(z.decode())
                    except Exception as e:
                        dictionary[column_names[index]].add(z)
                    index += 1
            except:
                break
    value = data_set.make_one_shot_iterator().get_next()
    default_values = []
    with tf.Session() as session:
        value = session.run(value)
        index = 0
        for z in value:
            try:
                float(z.decode())
                features.append(tf.feature_column.numeric_column(column_names[index]))
                default_values.append([0.0])
            except Exception as e:
                default_values.append([''])
                features.append(
                    tf.feature_column.categorical_column_with_hash_bucket(
                        column_names[index],
                        len(dictionary.get(column_names[index]))))

            index += 1
    return features, default_values
