#!/usr/bin/env python
# coding: utf-8

# In[41]:


from distutils.log import error
from logging import raiseExceptions
from msilib.schema import Error
from pyro import sample
import botorch
import numpy as np
import matplotlib.pyplot as plt
import torch

from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.generation import gen_candidates_scipy
from botorch.acquisition import ExpectedImprovement as EI 

from lmgp_pytorch.models.gpregression import GPR
from lmgp_pytorch.models.lmgp import LMGP

from botorch.models import SingleTaskGP
from lmgp_pytorch.optim.mll_scipy import fit_model_scipy

from botorch.sampling import SobolEngine
from botorch.utils.sampling import draw_sobol_samples

from scipy.optimize import minimize
from lmgp_pytorch.bayesian_optimizations.acquisition import EI_fun  

import pandas as pd
import random
from lmgp_pytorch.preprocessing.numericlevels import setlevels
from lmgp_pytorch.preprocessing import standard
import pickle
import time

def run_HOIP_Bayesian(seed, BO_initial = 5, qual_index = [], quant_index= list(range(3,10)), plotflag = False):

    ###############Parameters########################
    add_prior_flag = False
    num_minimize_init = 16
    predict_fidelity = 1
    level_set = None
    quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

    plt.rcParams['figure.dpi']=150
    plt.rcParams['font.family']='serif'
    #################################################
    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    set_seed(seed)
    ###################################################
    data = pd.read_csv('./datasets/sample_low.csv')
    X = data.iloc[:,:3].to_numpy()
    y = data.iloc[:,-1].to_numpy().astype(np.float64).reshape(-1,)
    y = torch.from_numpy(y)

    if level_set is None:
        level_set = [len(set(X[...,jj])) for jj in qual_index]

    ########### Make it standard ######################
    if len(qual_index) > 0:
        X = setlevels(X, qual_index)
        X = standard(X, quant_index)
    else:
        X = X[..., quant_index].astype(float)
        quant_index = list(range(len(quant_index)))
        X, mean_X, std_X = standard(X, quant_index)

    if len(quant_index) == 0:
        X = X[...,qual_index]
    
    ####################################################
    qual_index_lev = {i:j for i,j in zip(qual_index, level_set)}
    ########### Select few random choice ################
    random_index = np.random.randint(0, len(X)-2, BO_initial)
    Xtrain = X[random_index,:]
    ytrain = y[random_index]

    maximize_flag = False
    sign = 1


    t1 = time.time()

    ymin_list = []
    xmin_list = []

    cumulative_cost = []
    gain = []
    bestf = []

    if maximize_flag:
        best_f0 = ytrain.max().reshape(-1,) 
    else:
        best_f0 = ytrain.min().reshape(-1,) 

    ##########################################################################################################
    #-------------------------------------------Main Loop-----------------------------------------------------
    ##########################################################################################################
    n = 100
    for i in range(n):

        if maximize_flag:
            best_f = ytrain.max().reshape(-1,) 
            gain.append(best_f - best_f0)
            bestf.append(best_f)
        else:
            best_f = ytrain.min().reshape(-1,) 
            gain.append(best_f0 - best_f)
            bestf.append(best_f)

        ######################## Model ####################################    
        model = LMGP(
            train_x=Xtrain,
            train_y=ytrain,
            qual_ind_lev= qual_index_lev,
            quant_correlation_class= quant_kernel,
            NN_layers= [],
            fix_noise= False
        ).double()

        # optimize noise successively
        reslist,opt_history = fit_model_scipy(
            model, 
            num_restarts = num_minimize_init,
            add_prior=add_prior_flag, # number of restarts in the initial iteration
            n_jobs= -1
        )

        ####################################################################
        
        with torch.no_grad():
            ytest, ystd = model.predict(X, return_std=True)

        print(f'best f is {best_f}')

        scores = EI_fun(best_f, ytest.reshape(-1,1), ystd.reshape(-1,1), maximize = maximize_flag)

        print(f'This was the {i+1} iteration')
        #print(f'Batch_candidates = {batch_candidates}')
        #print(f'batch_acq_values = {batch_acq_values}')


        index = torch.argmax(scores)

        if index in random_index:
            Warning('Found the same Index')
            break

        random_index = np.append(random_index, index)
        Xnew = X[index,...]
        ynew = y[index]
        Xtrain = torch.cat([Xtrain,Xnew.reshape(1,-1)])
        ytrain = torch.cat([ytrain, ynew.view(1)])
        ymin_list.append(ynew.reshape(-1,))
        xmin_list.append(Xnew)
        
        print(f'Xnew is {Xnew} and \n ynew is {ynew}')
        #print(f'the best value of alpha for interval is {model.interval_alpha}')
        if all(Xnew == X[-1,:]):
            print('Found the best one')
            break


    plt.figure()
    plt.scatter(range(len(ymin_list)), ymin_list)

    plt.figure()
    plt.scatter(range(len(bestf)), bestf)
    plt.legend(f'Found the best one in {i} iteration')

    print(f'Total time is {time.time() - t1}')

    if plotflag:
        plt.show()

    if i ==0:
        return np.array([bestf[0],bestf[0]]), i, np.array([ymin_list[0],ymin_list[0]]), np.array([xmin_list[0], xmin_list[0]])

    return np.array(bestf), i, np.array(ymin_list), np.array(xmin_list)

if __name__ == '__main__':

######################################### LMGP_only_cat ################################################
    t1 = time.time()
    np.random.seed(12345)
    random_seed = np.random.choice(range(0,1000), size=1, replace=False)
    output = {'max_iteration':[], 'Best_value_found':[]}
    itr = 0
    for seed in random_seed:
        itr += 1
        [bestf, max_itr, ymin_list, xmin_list] = \
            run_HOIP_Bayesian(seed = seed, BO_initial = 100, qual_index = list(range(3)), quant_index= [], plotflag=False)
        print(f'***************** Random state: {itr} *****************')
        print(f'maximum iteration is {max_itr}')
        print(f'maximum index is {np.argmax(ymin_list)} for {ymin_list[np.argmax(ymin_list)]} at \n {xmin_list[np.argmax(ymin_list)]}')
        output['max_iteration'].append(max_itr)
        output['Best_value_found'].append(xmin_list[np.argmax(ymin_list)])
    
    file = open('LMGP_cat_initial_5', 'wb')
    pickle.dump(output, file)
    file.close()
    print(f'total time is {time.time() - t1}')
