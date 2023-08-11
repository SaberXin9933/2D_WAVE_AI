import numpy as np


def sigmodNP(x):
    return 1 / (1 + np.exp(-x))
