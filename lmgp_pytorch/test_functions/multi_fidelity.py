import numpy as np
import matplotlib.pyplot as plt
from pyro import param
from scipy.stats.qmc import Sobol, scale
import math
import torch


def wing_h(parameters=None, n=100):
    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if parameters is None:
        sobolset = Sobol(d=dx)
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        parameters = scale(parameters, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(parameters) != np.ndarray:
        parameters = np.array(parameters)
    Sw = parameters[..., 0]
    Wfw = parameters[..., 1]
    A = parameters[..., 2]
    Gama = parameters[..., 3] * (np.pi/180.0)
    q = parameters[..., 4]
    lamb = parameters[..., 5]
    tc = parameters[..., 6]
    Nz = parameters[..., 7]
    Wdg = parameters[..., 8]
    Wp = parameters[..., 9]
    # This is the output
    y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
        q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3) *\
        (Nz * Wdg) ** 0.49 + Sw * Wp

    if out_flag == 1:
        return parameters, y
    else:
        return y


def wing_l1(parameters=None, n=100):
    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if parameters is None:
        sobolset = Sobol(d=dx)
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        parameters = scale(parameters, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(parameters) != np.ndarray:
        parameters = np.array(parameters)
    Sw = parameters[..., 0]
    Wfw = parameters[..., 1]
    A = parameters[..., 2]
    Gama = parameters[..., 3] * (np.pi/180.0)
    q = parameters[..., 4]
    lamb = parameters[..., 5]
    tc = parameters[..., 6]
    Nz = parameters[..., 7]
    Wdg = parameters[..., 8]
    Wp = parameters[..., 9]
    # This is the output
    y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
        q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
        * (Nz * Wdg) ** 0.49 + 1 * Wp

    if out_flag == 1:
        return parameters, y
    else:
        return y


def wing_l2(parameters=None, n=100):
    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if parameters is None:
        sobolset = Sobol(d=dx)
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        parameters = scale(parameters, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(parameters) != np.ndarray:
        parameters = np.array(parameters)
    Sw = parameters[..., 0]
    Wfw = parameters[..., 1]
    A = parameters[..., 2]
    Gama = parameters[..., 3] * (np.pi/180.0)
    q = parameters[..., 4]
    lamb = parameters[..., 5]
    tc = parameters[..., 6]
    Nz = parameters[..., 7]
    Wdg = parameters[..., 8]
    Wp = parameters[..., 9]
    # This is the output
    y = 0.036 * Sw**0.8 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
        q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
        * (Nz * Wdg) ** 0.49 + 1 * Wp

    if out_flag == 1:
        return parameters, y
    else:
        return y


def wing_l3(parameters=None, n=100):
    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if parameters is None:
        sobolset = Sobol(d=dx)
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        parameters = scale(parameters, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(parameters) != np.ndarray:
        parameters = np.array(parameters)
    Sw = parameters[..., 0]
    Wfw = parameters[..., 1]
    A = parameters[..., 2]
    Gama = parameters[..., 3] * (np.pi/180.0)
    q = parameters[..., 4]
    lamb = parameters[..., 5]
    tc = parameters[..., 6]
    Nz = parameters[..., 7]
    Wdg = parameters[..., 8]
    Wp = parameters[..., 9]
    # This is the output
    y = 0.036 * Sw**0.9 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
        q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
        * (Nz * Wdg) ** 0.49 + 0 * Wp

    if out_flag == 1:
        return parameters, y
    else:
        return y


def multi_fidelity_wing(params={'h': 50, 'l1': 100, 'l2': 100, 'l3': 100}):
    n = sum([i for i in params.values()])
    X_list = []
    y_list = []
    for level, num in params.items():
        if level == 'h' and num > 0:
            X, y = wing_h(n=num)
            X = np.hstack([X, np.ones(num).reshape(-1, 1)])
            X_list.append(X)
            y_list.append(y)
        elif level == 'l1' and num > 0:
            X, y = wing_l1(n=num)
            X = np.hstack([X, np.ones(num).reshape(-1, 1) * 2])
            X_list.append(X)
            y_list.append(y)
        elif level == 'l2' and num > 0:
            X, y = wing_l2(n=num)
            X = np.hstack([X, np.ones(num).reshape(-1, 1) * 3])
            X_list.append(X)
            y_list.append(y)
        elif level == 'l3' and num > 0:
            X, y = wing_l3(n=num)
            X = np.hstack([X, np.ones(num).reshape(-1, 1) * 4])
            X_list.append(X)
            y_list.append(y)
        else:
            raise ValueError('Wrong label, should be h, l1, l2 or l3')
    return np.vstack([*X_list]), np.hstack(y_list)


def Augmented_braning(X):

    t1 = (
        X[..., 1]
        - (5.1 / (4 * math.pi ** 2) - 0.1 * (1 - X[..., 2])) * X[..., 0] ** 2
        + 5 / math.pi * X[..., 0]
        - 6
    )
    t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
    return t1 ** 2 + t2 + 10
