# Seasonality, Trends, Auto-correlation & Noise
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


def plot_series(time, series, style="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], style, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.rand(len(time)) * noise_level


def split_data():
    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]

    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return x_train, time_train, x_valid, time_valid


def auto_correlation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    phi1 = 0.5
    phi2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += phi1 * ar[step - 50]
        ar[step] += phi2 * ar[step - 33]
    return ar[50:] * amplitude


def autocorrelation(source, φs):
    ar = source.copy()
    max_lag = len(φs)
    for step, value in enumerate(source):
        for lag, φ in φs.items():
            if step - lag > 0:
                ar[step] += φ * ar[step - lag]
    return ar


def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series


if __name__ == '__main__':
    time = np.arange(4 * 365 + 1)
    # Trend plot
    series = trend(time, 0.1)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    # Seasonality plot
    baseline = 10
    amplitude = 40

    series = baseline + seasonality(time, period=365, amplitude=amplitude)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    # Trend and Seasonality
    slope = 0.05
    series = baseline + trend(time, slope=slope) + seasonality(time, period=365, amplitude=amplitude)

    # Noise
    noise_level = 5
    noise = white_noise(time, noise_level, seed=1)
    plt.figure(figsize=(10, 6))
    plot_series(time, noise)

    series = series + noise
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    # Auto-correlation
    series = auto_correlation(time, 10, seed=1)
    plt.figure(figsize=(10, 6))
    plot_series(time[:200], series[:200])

    # Auto-correlation with trend
    series = auto_correlation(time, 10, seed=42) + trend(time, 2)
    plt.figure(figsize=(10, 6))
    plot_series(time[:200], series[:200])

    # Auto-correlation with seasonality and trend
    series = auto_correlation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
    plt.figure(figsize=(10, 6))
    plot_series(time[:200], series[:200])

    # Impulses
    series = impulses(time, 10, seed=42)
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    signal = impulses(time, 10, seed=42)
    series = autocorrelation(signal, {1: 0.99})
    plot_series(time, series)
    plt.plot(time, signal, "k-")

    signal = impulses(time, 10, seed=42)
    series = autocorrelation(signal, {1: 0.70, 50: 0.2})
    plot_series(time, series)
    plt.plot(time, signal, "k-")
    plt.show()

    series_diff1 = series[1:] - series[:-1]
    plot_series(time[1:], series_diff1)

    # Pandas auto-correlation
    autocorrelation_plot(series)

    # ARIMA
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    plt.show()
