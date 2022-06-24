import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol, scale
import math
import torch


####################################Wing Function################################################

def wing(n=100, X = None, noise_std = 0.0, random_state = None, shuffle = True):
    """_summary_

    Args:
        parameters (_type_, optional): For evaluation, you can give parameters and get the values. Defaults to None.
        n (int, optional): defines the number of data needed. Defaults to 100.

    Returns:
        _type_: if paramters are given, it returns y, otherwise it returns both X and y
    """

    if random_state is not None:
        np.random.seed(random_state)

    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(X) != np.ndarray:
        X = np.array(X)
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (np.pi/180.0)
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    # This is the output
    y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
        q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3) *\
        (Nz * Wdg) ** 0.49 + Sw * Wp


    if shuffle:
        index = np.random.randint(0, len(y), size = len(y))
        X = X[index,...]
        y = y[index]


    if noise_std > 0.0:
        if out_flag == 1:
            return X, y + np.random.randn(*y.shape) * noise_std
        else:
            return y       
    else:
        if out_flag == 1:
            return X, y
        else:
            return y


####################################Borehole Function################################################

def Borehole(n=100, X = None, noise_std = 0.0, random_state = None, shuffle = True):
    """_summary_

    Args:
        parameters (_type_, optional): For evaluation, you can give parameters and get the values. Defaults to None.
        n (int, optional): defines the number of data needed. Defaults to 100.

    Returns:
        _type_: if paramters are given, it returns y, otherwise it returns both X and y
    """

    if random_state is not None:
        np.random.seed(random_state)

    dx = 8
    l_bound = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
    u_bound = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
    out_flag = 0
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(X) != np.ndarray:
        X = np.array(X)
    rw = X[..., 0]
    r = X[..., 1]
    Tu = X[..., 2]
    Hu = X[..., 3] * (np.pi/180.0)
    Tl = X[..., 4]
    Hl = X[..., 5]
    L = X[..., 6]
    Kw = X[..., 7]

    frac1 = 2 * np.pi * Tu * (Hu-Hl)

    frac2a = 2*L*Tu / (np.log(r/rw)*rw**2*Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r/rw) * (1+frac2a+frac2b)

    y = frac1 / frac2


    if shuffle:
        index = np.random.randint(0, len(y), size = len(y))
        X = X[index,...]
        y = y[index]


    if noise_std > 0.0:
        if out_flag == 1:
            return X, y + np.random.randn(*y.shape) * noise_std
        else:
            return y       
    else:
        if out_flag == 1:
            return X, y
        else:
            return y


####################################OTL CIRCUIT Function################################################


####################################PistonFunction######################################################



#########################################################################################################