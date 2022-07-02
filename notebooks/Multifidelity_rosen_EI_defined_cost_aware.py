#!/usr/bin/env python
# coding: utf-8

# In[41]:


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

from Rosenbrock import generate_rosen, rosen, rosen_low


###############Parameters########################
add_prior_flag = True
num_minimize_init = 24
qual_index = [2]
quant_index= [0,1]
level_sets = [0,1]
predict_fidelity = 1

quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'
#################################################
noise_indices=[] # if you make this [], then you will have one single noise for all cases []
Optimization_technique='L-BFGS-B' #optimization methods: 'BFGS' 'L-BFGS-B' 'Newton-CG' 'trust-constr'  'SLSQP'
Optimization_constraint=False   # False # True 
regularization_parameter=[0, 0] ###### Always at least one element of regularization_parameter should be zero: The first element is for L1 regularization and the second element is for L2
Bounds=True


################################################

n_high_fidelity = 5
n_low_fidelity = 1
low_b, high_b = 0.0, 1.0  # this is (0.,0), (1,1) and (0,1) for high, low and mix
n = 30
cost_fun = lambda x:1000 if x==0 else 1
###################################################

np.random.seed(10)
data = generate_rosen(plot_flag=0)
high = data[np.random.choice(range(0,10000), n_high_fidelity), :]
low = data[np.random.choice(range(10000,20000), n_low_fidelity), :]

train_data = np.vstack([high, low])

X_init = train_data[:,[0,1,3]]

# X_range is [0, 1], therefore we can get the reponse directly  
# from the objective function
# Get the initial responses
Y_init = train_data[:,[2]]


X_init = X_init.astype(np.float64)
Y_init = Y_init.astype(np.float64)

X = data[:,[0,1,3]]
y = data[:,[2]]

Xtrain = torch.from_numpy(X_init)

ytrain = torch.from_numpy(Y_init)
ytrain = ytrain.reshape(-1,)


maximize_flag = False
sign = 1


def EI_fun(best_f, mean, std, maximize = True, si = 0.0, cost = None):
    from torch.distributions import Normal

    # deal with batch evaluation and broadcasting
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    u = (mean - best_f.expand_as(mean) - torch.sign(best_f) * si) / sigma
    
    if cost is None:
        cost = torch.ones(u.shape)    

    cost = cost.view(u.shape)

    if not maximize:
        u = -u
        #si = -si
    
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei/cost


import time
t1 = time.time()

ymin_list = []
xmin_list = []

initial_cost = np.sum(list(map(cost_fun,Xtrain[:,-1])))
cumulative_cost = []
gain = []
bestf = []

if maximize_flag:
    best_f0 = ytrain.max().reshape(-1,) 
else:
    best_f0 = ytrain.min().reshape(-1,) 



for i in range(n):


    if maximize_flag:
        best_f = ytrain.max().reshape(-1,) 
        gain.append(best_f - best_f0)
        bestf.append(best_f)
    else:
        best_fh = ytrain[Xtrain[:,-1] == 0].min().reshape(-1,) 
        best_fl = ytrain[Xtrain[:,-1] == 1].min().reshape(-1,) 
        gain.append(best_f0 - best_fh)
        bestf.append(best_fh)

    ######################## Model ####################################    
    model = LMGP(
        train_x=Xtrain,
        train_y=ytrain,
        noise_indices=noise_indices,
        qual_index= qual_index,
        quant_index= quant_index,
        num_levels_per_var= level_sets,
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

    ####################################################################
    
    # Bound is defined as min max of all dimensions. [[min1, min2, min3, ...], [max1, max2, max3, ...]]
    
    low_b = 0
    high_b = 0
    bounds = torch.tensor([[-2.0, -2.0, low_b], [2.0, 2.0, high_b]])
    samples = draw_sobol_samples(bounds,n=500,q = 1, batch_shape= None, seed=12345)
    samples = samples.squeeze(1).double()
    samples[:,-1] = torch.round(samples[:,-1])

    total = samples.clone()

    with torch.no_grad():
        ytesth, ystdh = model.predict(samples, return_std=True)

    costh = torch.tensor(list(map(cost_fun, samples[:,-1])))

    low_b = 1
    high_b = 1
    bounds = torch.tensor([[-2.0, -2.0, low_b], [2.0, 2.0, high_b]])
    samples = draw_sobol_samples(bounds,n=500,q = 1, batch_shape= None, seed=12345)
    samples = samples.squeeze(1).double()
    samples[:,-1] = torch.round(samples[:,-1])

    with torch.no_grad():
        ytestl, ystdl = model.predict(samples, return_std=True)

    costl = torch.tensor(list(map(cost_fun, samples[:,-1])))


    print(f'best f is {best_fh}')


    scoresh = EI_fun(best_fh, ytesth.reshape(-1,1), ystdh.reshape(-1,1), maximize = maximize_flag, cost = costh, si = 0.0)


    scoresl = EI_fun(best_fl, ytestl.reshape(-1,1), ystdl.reshape(-1,1), maximize = maximize_flag, cost = costl, si = torch.abs(0.01*best_fl))

    scores = torch.cat([scoresh, scoresl])

    samples = torch.cat([total, samples], axis = 0)

    print(f'This was the {i+1} iteration')
    #print(f'Batch_candidates = {batch_candidates}')
    #print(f'batch_acq_values = {batch_acq_values}')


    index = torch.argmax(scores)
    Xnew = samples[index,...]
    Xtrain = torch.cat([Xtrain,Xnew.reshape(1,-1)])
    if Xnew[-1] == 1:
        ynew = sign * rosen_low([Xnew[0].numpy(), Xnew[1].numpy()])
    elif Xnew[-1] == 0:
        ynew = sign * rosen([Xnew[0].numpy(), Xnew[1].numpy()])

    ytrain = torch.cat([ytrain, torch.from_numpy(ynew.reshape(-1,))])

    ymin_list.append(ynew.reshape(-1,))
    xmin_list.append(Xnew)
    
    #-----------------------------------------
    cumulative_cost.append(initial_cost + cost_fun(Xnew[-1]))

    initial_cost = cumulative_cost[-1]

    print(f'Xnew is {Xnew} and ynew is {ynew}')

    if i%5000 == 0:

        plt.figure()
        ax = plt.axes(projection = '3d')
        #plt.plot(X,y)
        ax.scatter(Xtrain[:,0], Xtrain[:,1], ytrain, color = 'green')
        ax.scatter(Xnew[0],Xnew[1], ynew, color = 'red')
        

        # plt.figure()
        # ax = plt.axes(projection = '3d')
        # ax.scatter(samples[:,0], samples[:,1], EI_fun(best_f, ytest.reshape(-1,1), ystd.reshape(-1,1), maximize=maximize_flag).detach())
        # ax.scatter(Xnew[0],Xnew[1], scores[index], color = 'red')
        #plt.show()
    
    


plt.figure()
plt.scatter(range(n), ymin_list)

plt.figure()
plt.plot(cumulative_cost, gain, 'b.-')

plt.figure()
plt.plot(cumulative_cost, np.array(bestf) * -1, 'b.-')
plt.show()

print(f'Total time is {time.time() - t1}')
index_min = np.argmin(np.array(ymin_list))
print(f'The minium found is {ymin_list[index_min]} at {xmin_list[index_min]}')

hfid_index = np.argmin(ytrain[Xtrain[:,-1] == 0])
print(f'The minium found in high fidelity {ytrain[Xtrain[:,-1] == 0][hfid_index]} at {Xtrain[Xtrain[:,-1] == 0][hfid_index]}')