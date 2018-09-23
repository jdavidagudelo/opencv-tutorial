import tensorflow as tf


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
