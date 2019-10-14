import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

# https://www.kaggle.com/robervalt/sunspots#Sunspots.csv

file_path = '../data/sunspots/sunspots.csv'


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def get_data(file_path):
    sunspots = []
    time_steps = []

    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)

        for row in reader:
            sunspots.append(float(row[2]))
            time_steps.append(int(row[0]))

    return np.array(sunspots), np.array(time_steps)


def model_forecast(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    return forecast


def windowed_dataset_dnn(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))
    return dataset.batch(batch_size).prefetch(1)


def windowed_dataset_cnn(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))
    return dataset.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    return model.predict(dataset)


def dnn_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(20, input_shape=[window_size], activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])


def cnn_lstm_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=60, kernel_size=5, strides=1, padding='causal', activation='relu',
                               input_shape=[None, 1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(60, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(60, return_sequences=True)),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400),
    ])


def train_dnn():
    train_set = windowed_dataset_dnn(series=x_train,
                                     window_size=window_size,
                                     batch_size=batch_size,
                                     shuffle_buffer=shuffle_buffer_size)
    model = dnn_model()
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9))
    model.fit(train_set, epochs=1, verbose=0)

    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
    forecast = forecast[split_time - window_size:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))
    plot_series(time_val, x_val)
    plot_series(time_val, results)

    print(tf.keras.metrics.mean_absolute_error(x_val, results).numpy())


def train_cnn_lstm():
    train_set = windowed_dataset_cnn(series=x_train,
                                     window_size=window_size,
                                     batch_size=batch_size,
                                     shuffle_buffer=shuffle_buffer_size)
    model = cnn_lstm_model()
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9))
    history = model.fit(train_set, epochs=5, verbose=0)

    forecast = []
    forecast = model_forecast(model, series[..., np.newaxis], window_size, batch_size)
    results = forecast[split_time - window_size:-1, -1, 0]

    plt.figure(figsize=(10, 6))
    plot_series(time_val, x_val)
    plot_series(time_val, results)
    print(tf.keras.metrics.mean_absolute_error(x_val, results).numpy())

    # Retrieve a list of list results on training and test data sets for each training epoch
    loss = history.history['loss']
    epochs = range(len(loss))  # Get number of epochs

    # Plot training and validation loss per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])

    # Plot training and validation loss per epoch
    zoomed_loss = loss[0:]
    zoomed_epochs = range(0, 5)
    plt.figure(figsize=(10, 6))
    plt.plot(zoomed_epochs, zoomed_loss, 'r')
    plt.title('Zoomed Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])


if __name__ == '__main__':
    series, time = get_data(file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(time, series)

    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]

    time_val = time[split_time:]
    x_val = series[split_time:]

    window_size = 60
    batch_size = 32
    shuffle_buffer_size = 1000

    tf.keras.backend.clear_session()
    tf.random.set_seed(52)
    np.random.seed(52)

    # DNN
    train_dnn()

    # CNN + LSTM
    train_cnn_lstm()

    plt.show()
