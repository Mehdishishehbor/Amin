from lmgp_pytorch.models.lmgp import LMGP
from lmgp_pytorch.optim.mll_scipy import fit_model_scipy
# from lmgp_pytorch.bayesian_optimizations.acquisition import EI_cost_aware  
import pandas as pd
import random
from lmgp_pytorch.preprocessing.numericlevels import setlevels
from lmgp_pytorch.preprocessing import standard
import pickle
import torch
import time
import os
import random
t1 = time.time()
import numpy as np
import matplotlib.pyplot as plt
from lmgp_pytorch.test_functions.multi_fidelity import  multi_fidelity_wing_value
from botorch.utils.sampling import manual_seed
from torch import Tensor
from botorch.utils.sampling import draw_sobol_samples
from scipy.optimize import minimize


def run_MA2X_Bayesian(seed, plotflag = False):

    #################################################
    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    set_seed(seed)


####################values to be defined##############################
    num_optimization_samples=20000
    add_prior_flag = False
    num_minimize_init = 12
    qual_index = [10]
    quant_index= list(range(10))
    num_BO_iteration= 50
    level_set= [4]
    quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'
    max_cost=40000
    num_high=5
    num_low=10
    noise_indices=[]
    Optimization_technique='L-BFGS-B'
    Optimization_constraint=False 
    regularization_parameter=[0, 0]
    Bounds=True

########################Cost function################################

    def cost_fun(x):
        if x==0:
            return 1000
        elif x==1:
            return 100
        elif x==2:
            return 10
        elif x==3:
            return 1
        
####################################Data######################################

    tkwargs = {"dtype": torch.double,
        "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),}
    problem = lambda x: multi_fidelity_wing_value(x).to(**tkwargs)

    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025,0]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08,level_set[0]-1]
    bounds = torch.tensor([l_bound, u_bound], **tkwargs)

    fidelities = torch.tensor(sorted(random.sample(range(l_bound[-1], u_bound[-1]+1), u_bound[-1]+1)), **tkwargs)
    
    
    def generate_initial_data(num_high=0, num_low=0):
        # generate training data
        if num_low ==0:
            train_x_high = draw_sobol_samples(bounds[:,:-1],n=num_high,q = 1, batch_shape= None).squeeze(1).to(**tkwargs)
            train_f_high = fidelities[torch.randint(1, (num_high, 1))]
            train_x_full = torch.cat((train_x_high, train_f_high), dim=1)
            train_obj = problem(train_x_full).unsqueeze(-1) 
        
        else:
            train_x_high = draw_sobol_samples(bounds[:,:-1],n=num_high,q = 1, batch_shape= None).squeeze(1).to(**tkwargs)
            train_f_high = fidelities[torch.randint(1, (num_high, 1))]
            train_x_full_high = torch.cat((train_x_high, train_f_high), dim=1)
            train_x_low = draw_sobol_samples(bounds[:,:-1],n=num_low,q = 1, batch_shape= None).squeeze(1).to(**tkwargs)
            train_f_low = fidelities[(torch.randint(level_set[-1]-1, (num_low, 1)))+1]
            train_x_full_low = torch.cat((train_x_low, train_f_low), dim=1)
            train_x_full = torch.cat((train_x_full_high, train_x_full_low), dim=0)
            train_obj = problem(train_x_full).unsqueeze(-1) 


        return train_x_full, train_obj


    X, y = generate_initial_data(num_high=num_high, num_low=num_low)
    

    if level_set is None:
        level_set = [len(set(X[...,jj])) for jj in qual_index]
   
    y = y.reshape(-1)
    #######generate samples for high#########################
 

    ########### Make it standard ######################
    if len(qual_index) == 0:
        X=X[...,quant_index]
        samples=samples[...,quant_index]

    X,xmean, xstd = standard(X, quant_index)


    if len(quant_index) == 0:
        X = X[...,qual_index]

    Xtrain = X
    ytrain = y

#############################################################
    plt.rcParams['figure.dpi']=150
    plt.rcParams['font.family']='serif'
    

############# Define maximization or minimization###################
    maximize_flag = False
    sign = 1

###################Assign cost ##################
    initial_cost = np.sum(list(map(cost_fun,Xtrain[:,-1])))
##################### Results lists #####################
    t1 = time.time()
    ymin_list = []
    xmin_list = []   
    cumulative_cost = []
    bestf = []
    best_x=[]
    Fidelity=[]



    if maximize_flag:
        best_fh = ytrain[Xtrain[:,-1] == 0].max().reshape(-1,) 
        bestf.append(np.round(best_fh,2))
    else:
        best_fh0 = ytrain[Xtrain[:,-1] == 0].min().reshape(-1,)
        bestf.append(np.round(best_fh0[0],2))

    cumulative_cost.append(initial_cost)
        


    ##########################################################################################################
    #-------------------------------------------Main Loop-----------------------------------------------------
    ##########################################################################################################
    
    
    i=0
    while cumulative_cost[-1] < max_cost:

        i+=1

        if maximize_flag:
            best_fh = ytrain[Xtrain[:,-1] == 0].max().reshape(-1,) 
            best_fl = ytrain[Xtrain[:,-1] != 0].max().reshape(-1,) 
            bestf.append(np.round(best_fh,2))
        else:
            best_fh = ytrain[Xtrain[:,-1] == 0].min().reshape(-1,)
            best_fl1 = ytrain[Xtrain[:,-1] == 1].min().reshape(-1,) 
            best_fl2 = ytrain[Xtrain[:,-1] == 2].min().reshape(-1,) 
            best_fl3 = ytrain[Xtrain[:,-1] == 3].min().reshape(-1,) 
            best_x0=Xtrain[Xtrain[:,-1] == 0][np.argmin(ytrain[Xtrain[:,-1] == 0])]
            bestf.append(np.round(best_fh[0],2))
            best_x.append(best_x0)

        
        if len(bestf)> 100:
            if np.var(bestf[-50:])<1e-6:
                break
        

        ######################## Model ####################################    
        model = LMGP(
        train_x=Xtrain,
        train_y=ytrain,
        noise_indices=noise_indices,
        qual_index= qual_index,
        quant_index= quant_index,
        num_levels_per_var= level_set,
        quant_correlation_class= quant_kernel,
        NN_layers= [],
        fix_noise= False,
        lb_noise = 1e-20
        ).double()

        LMGP.reset_parameters

        # optimize noise successively
        reslist,opt_history = fit_model_scipy(
            model, 
            num_restarts = num_minimize_init,
            add_prior=add_prior_flag, # number of restarts in the initial iteration
            n_jobs= 1,
            method = Optimization_technique,
            constraint=Optimization_constraint,
            regularization_parameter=regularization_parameter,
            bounds=Bounds,
    )

        ################### Functions ############################

        def EI_cost(samples,best_f,model=model, maximize = False, si = 0.0):
            from torch.distributions import Normal
            #samples=torch.from_numpy(samples.reshape(1,-1))
            samples=torch.tensor(samples.reshape(1,-1))

            with torch.no_grad():
                mean, std = model.predict(samples, return_std=True)

            cost = torch.tensor(list(map(cost_fun, torch.tensor(samples[:,-1], dtype = torch.int64))))

            mean=mean.reshape(-1,1)

            # deal with batch evaluation and broadcasting
            view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
            mean = mean.view(view_shape)
            sigma = std.view(view_shape)
            u = (mean - best_f - np.sign(best_f) * si) / sigma
            
            if cost is None:
                cost = torch.ones(u.shape)    

            cost = cost.view(u.shape)

            if not maximize:
                u = -u
                #si = -si
            
            normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            ei = sigma * updf
            return -1 * (ei/cost)


        def EI_cost_high(samples,best_f=best_fh, model=model, maximize = False, si = 0.0):
            from torch.distributions import Normal
            samples=torch.tensor(samples.reshape(1,-1))
            with torch.no_grad():
                mean, std = model.predict(samples, return_std=True)

            mean=mean.reshape(-1,1)

            cost = torch.tensor(list(map(cost_fun, torch.tensor(samples[:,-1], dtype = torch.int64))))

            # deal with batch evaluation and broadcasting
            view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
            mean = mean.view(view_shape)
            sigma = std.view(view_shape)
            u = (mean - best_f - np.sign(best_f) * si) / sigma
            
            if cost is None:
                cost = torch.ones(u.shape)    

            cost = cost.view(u.shape)

            if not maximize:
                u = -u
                #si = -si
            
            normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            # ei = sigma * u * ucdf
            ei= sigma * (updf + u * ucdf)
            return -1* (ei/cost)
        
       
        ################# Scores ######################### 
    


        X_list=[]   
        y_list=[] 

        bound_h=((150,200),(270,300),(6,10),(-10,10),(15,45),(0.5,1),(0.08,0.18),(2.5,6),(1700,2500),(0.025,0.08),(0,0))
        bound_l1=((150,200),(270,300),(6,10),(-10,10),(15,45),(0.5,1),(0.08,0.18),(2.5,6),(1700,2500),(0.025,0.08),(1,1))
        bound_l2=((150,200),(270,300),(6,10),(-10,10),(15,45),(0.5,1),(0.08,0.18),(2.5,6),(1700,2500),(0.025,0.08),(2,2))
        bound_l3=((150,200),(270,300),(6,10),(-10,10),(15,45),(0.5,1),(0.08,0.18),(2.5,6),(1700,2500),(0.025,0.08),(3,3))
        

        def run_scipy(EI, best_f, bound, model, fidelity):
            temp = np.empty((12,))
            tempx = []
            for i in range(12):
                ################ High ###########################
                l_bound_h = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025,fidelity]
                u_bound_h = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08,fidelity]
                bounds_h = torch.tensor([l_bound_h, u_bound_h], **tkwargs)
                samples_h = draw_sobol_samples(bounds_h,n=1,q = 1, batch_shape= None)
                samples_h = samples_h.squeeze(1).double()
                samples_h[:,-1] = torch.round(samples_h[:,-1])
                samples_h= torch.cat([((samples_h[0][0:-1]-xmean)/xstd).reshape(1,-1), samples_h[0][-1].reshape(-1,1)],dim=-1)
                result_h = minimize(EI,samples_h.reshape(-1,), args=(best_f, model),bounds=bound)
                temp[i] = result_h.fun
                tempx.append(result_h.x)
            min_index = np.argmin(temp)
            Y_h = temp[min_index]
            X_h = tempx[min_index]
            return Y_h, X_h
        

        #################################################
        Y_h, X_h = run_scipy(EI_cost_high, best_fh, bound_h, model, fidelity = 0)
        Y_l1, X_l1 = run_scipy(EI_cost, best_fl1, bound_l1, model, fidelity = 1)
        Y_l2, X_l2 = run_scipy(EI_cost, best_fl2, bound_l2, model,fidelity = 2)
        Y_l3, X_l3 = run_scipy(EI_cost, best_fl3, bound_l3, model, fidelity = 3)
            ##########################################
        y_list=[Y_h,Y_l1,Y_l2,Y_l3]
        X_list=[X_h,X_l1,X_l2,X_l3]
        print('############## Found EI/cost #################')
        print(y_list)
        print('##############################################')

        min_index = np.argmin(y_list)
        Xnew = X_list[min_index]
        ynew = torch.tensor(problem(torch.tensor(Xnew).unsqueeze(0)))
        ############# Total calculation ###################

        print(f'This was the {i+1} iteration')
        
        # temp=torch.tensor(temp)
        # Xnew= torch.cat([((temp[0:-1]-xmean)/xstd).reshape(1,-1), temp[-1].reshape(-1,1)],dim=-1)
        Xtrain = torch.cat([Xtrain,torch.tensor(Xnew.reshape(1,-1))])
        ytrain = torch.cat([ytrain, ynew.reshape(-1,)])
        ymin_list.append(ynew.reshape(-1,))
        xmin_list.append(Xnew)
        ##################
        cumulative_cost.append(initial_cost + cost_fun(Xnew[-1]))
        initial_cost = cumulative_cost[-1]
        Fidelity.append(Xnew[-1])
        
    
        print(f'Xnew is {Xnew} and \n ynew is {ynew}')
        print(f'Best_f is {best_fh}')
        

    print(f'Total time is {time.time() - t1}')

    
    return np.array(bestf), i, np.array(ymin_list), np.array(xmin_list),np.array(cumulative_cost), np.array(best_x) ,np.array(Fidelity)


if __name__ == '__main__':
    t1 = time.time()
    np.random.seed(12345)
    random_seed = np.random.choice(range(0,1000), size=20, replace=False)
    output = {'cost':[],'best_f':[],'best_x':[],'Fidelity':[]}
    itr = 0
    for seed in random_seed:
        itr += 1
        [bestf, max_itr, ymin_list, xmin_list,cost,best_x,Fidelity] = run_MA2X_Bayesian(seed = seed, plotflag=False)
        print(f'***************** Random state: {itr} *****************')
        print(f'maximum iteration is {max_itr}')
        print(f'maximum index is {np.argmin(ymin_list)} for {ymin_list[np.argmin(ymin_list)]} at \n {xmin_list[np.argmin(ymin_list)]}')
        output['cost'].append(cost)
        output['best_f'].append(bestf)
        output['best_x'].append(best_x)
        output['Fidelity'].append(Fidelity)
        


        
  

    # file = open("D:/Research/codes/BO_Gpyorch/New_approach/Final_codes/Results/Wing_CA_EI_LMGP.pkl", "wb")
    # pickle. dump(output,file)
    # file. close()
   
    
