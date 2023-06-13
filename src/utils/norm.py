import numpy as np


def normalize(data, std, mean, eps=1e-8):
    return (data - mean[np.newaxis, :]) / std[np.newaxis, :]


def renormalize(data, std, mean, eps=1e-8):
    return data * std[np.newaxis, :] + mean[np.newaxis, :]
