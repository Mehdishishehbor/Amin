import numpy as np
import matplotlib.pyplot as plt
from pyro import param
from scipy.stats.qmc import Sobol, scale


def wing_h(parameters=None, n=100):
    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if parameters is None:
        sobolset = Sobol(d=dx)
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n,:]
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
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n,:]
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
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n,:]
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
        parameters = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n,:]
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