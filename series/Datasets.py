import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def dataset_range():
    dataset = tf.data.Dataset.range(10)
    for val in dataset:
        print(val.numpy())


def dataset_window():
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(size=5, shift=1)
    for window_dataset in dataset:
        for val in window_dataset:
            print(val.numpy(), end=" ")
        print()


def dataset_drop_window():
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(size=5, shift=1, drop_remainder=True)
    for window_dataset in dataset:
        for val in window_dataset:
            print(val.numpy(), end=" ")
        print()


def dataset_flat_map_window():
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(size=5, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))
    for window in dataset:
        print(window.numpy())


def dataset_features_window():
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(size=5, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.shuffle(buffer_size=10)
    for x, y in dataset:
        print(x.numpy(), y.numpy())


def dataset_features_batch():
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(size=5, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size=2).prefetch(1)
    for x, y in dataset:
        print(x.numpy(), y.numpy())


if __name__ == '__main__':
    # dataset_range()
    # dataset_window()
    # dataset_drop_window()
    # dataset_flat_map_window()
    # dataset_features_window()
    dataset_features_batch()
