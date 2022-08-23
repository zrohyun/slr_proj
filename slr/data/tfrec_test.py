import tensorflow as tf


def get_ksl_dataset_from_tfrec(file, comp, batch_size, channel=2):
    dataset = tf.data.TFRecordDataset(file, comp)
    parsed_dataset = get_parsed_dataset(dataset, batch_size, channel)
    parsed_dataset = parsed_dataset.as_numpy_iterator()
    # tfds.as_numpy(parsed_dataset)
    return parsed_dataset


def get_parsed_dataset(tfrec_Dset: tf.data.TFRecordDataset, batch_size, channel):
    image_feature_description = {
        "raw_data": tf.io.FixedLenFeature([], tf.string),
        "data_shape": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example):
        example = tf.io.parse_single_example(example, image_feature_description)
        # example['label'] = tf.cast(example['label'],tf.string)
        # example['label'] = example['label']
        # raise Exception(f'{type(example['raw_data'])}')
        example["raw_data"] = tf.reshape(
            tf.io.decode_raw(example["raw_data"], tf.float32), (-1, 137, 3)
        )
        # session = tf.Session()
        # session.run()
        print("hello")
        print(example["label"])
        # tf.py_function(lambda x: int(x.decode()), example['label'],tf.string)
        return {"raw_data": example["raw_data"], "label": example["label"]}
        # raise Exception(f'{dir(example["label"])[-20:]},{example["label"]}')
        # return {'raw_data':example['raw_data'], 'label':tf.io.decode_raw(example['label'],tf.int64)}

    # session.run(my_example, feed_dict={serialized: my_example_str})
    parsed_dataset = tfrec_Dset.map(_parse_image_function).batch(batch_size)
    parsed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return parsed_dataset


if __name__ == "__main__":
    for i in get_ksl_dataset_from_tfrec(
        "/Users/0hyun/Downloads/gzip_test.tfrec", "GZIP", 1
    ):
        print(i)
        break
