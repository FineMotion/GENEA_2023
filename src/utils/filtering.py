from math import pi, sqrt, tan
import numpy as np


def butter(signal: np.array, delta_time_in_sec: float = 1.0 / 30, cut_off: float = 0.5 * 13 / 2):
    """
    literal translation for C# Butterworth filter from AI4Animation
    :param signal: input 1D signal
    :param delta_time_in_sec: duration of one frame
    :param cut_off: maximum frequency parameter, taken from quadruped initialization: pastKeys = futureKeys = 6
    :return: smoothed signal
    """
    df2 = signal.shape[0] - 1
    dat2 = np.zeros(df2 + 4)
    dat2[2:df2 + 2] = signal[:-1]
    dat2[1] = dat2[0] = signal[0]
    dat2[df2 + 3] = dat2[df2 + 2] = signal[-1]
    wc = tan(cut_off * pi * delta_time_in_sec)
    k1 = sqrt(2) * wc
    k2 = wc * wc
    a = k2 / (1 + k1 + k2)
    b = 2 * a
    c = a
    k3 = b / k2
    d = -2 * a + k3
    e = 1 - (2 * a) - k3

    dat_yt = np.zeros(df2 + 4)
    dat_yt[1] = dat_yt[0] = signal[0]
    for s in range(2, df2 + 2):
        dat_yt[s] = a * dat2[s] + b * dat2[s - 1] + c * dat2[s - 2] \
                    + d * dat_yt[s - 1] + e * dat_yt[s - 2]
    dat_yt[df2 + 3] = dat_yt[df2 + 2] = dat_yt[df2 + 1]
    dat_zt = np.zeros(df2 + 2)
    dat_zt[df2] = dat_yt[df2 + 2]
    dat_zt[df2 + 1] = dat_yt[df2 + 3]
    for t in range(-df2 + 1, 0):
        dat_zt[-t] = a * dat_yt[-t + 2] + b * dat_yt[-t + 3] + c * dat_yt[-t + 4] \
                     + d * dat_zt[-t + 1] + e * dat_zt[-t + 2]
    return dat_zt[:df2 + 1]
