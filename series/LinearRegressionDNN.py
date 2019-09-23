import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras.api._v1.keras as keras


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size=batch_size).prefetch(1)
    return dataset


if __name__ == '__main__':
    time = np.arange(4 * 365 + 1, dtype="float32")

    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5

    # Build time series
    series = (baseline +
              trend(time=time, slope=slope) +
              seasonality(time=time, period=365, amplitude=amplitude))
    series += noise(time=time, noise_level=noise_level, seed=1)

    # Data for train and eval
    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]

    time_val = time[split_time:]
    x_val = series[split_time:]

    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000

    dataset = windowed_dataset(series=x_train,
                               window_size=window_size,
                               batch_size=batch_size,
                               shuffle_buffer=shuffle_buffer_size)
    print(dataset)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20))

    model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9))
    history = model.fit(dataset, epochs=5, callbacks=[lr_schedule], verbose=0)

    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time - window_size:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))
    plot_series(time_val, x_val)
    plot_series(time_val, results)
    plt.show()

    print(keras.metrics.mean_absolute_error(x_val, results).numpy())

    lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    plt.semilogx(lrs, history.history["loss"])
    plt.axis([1e-8, 1e-3, 0, 300])

    window_size = 30
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9)
    model.compile(loss="mse", optimizer=optimizer)
    history = model.fit(dataset, epochs=500, verbose=0)

    loss = history.history['loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.show()

    # Plot all but the first 10
    loss = history.history['loss']
    epochs = range(10, len(loss))
    plot_loss = loss[10:]
    print(plot_loss)
    plt.plot(epochs, plot_loss, 'b', label='Training Loss')
    plt.show()

    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time - window_size:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))

    plot_series(time_val, x_val)
    plot_series(time_val, results)

    tf.keras.metrics.mean_absolute_error(x_val, results).numpy()