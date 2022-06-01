
import numpy as np
import torch


def standard(Xtrain, quant_index, Xtest = None):

    mean_xtrain =  Xtrain[...,quant_index].mean(axis = 0)
    std_xtrain =  Xtrain[...,quant_index].std(axis = 0)
    Xtrain[...,quant_index] = (Xtrain[..., quant_index] - mean_xtrain )/std_xtrain
    if Xtest is None:
        return Xtrain
    else:
        Xtest[...,quant_index] = (Xtest[..., quant_index] - mean_xtrain )/std_xtrain
        return Xtrain, Xtest

