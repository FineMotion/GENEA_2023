import numpy as np


def normalize(data, std, mean):
    return (data - mean[np.newaxis, :]) / std[np.newaxis, :]


def renormalize(data, std, mean):
    return data * std[np.newaxis, :] + mean[np.newaxis, :]