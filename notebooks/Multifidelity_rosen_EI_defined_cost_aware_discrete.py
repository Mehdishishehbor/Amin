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
add_prior_flag = False
num_minimize_init = 7
qual_index = [2]
quant_index= [0,1]
level_sets = [0,1]
predict_fidelity = 1

quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'
#################################################

n_high_fidelity = 5
n_low_fidelity = 5
low_b, high_b = 0.0, 1.0  # this is (0.,0), (1,1) and (0,1) for high, low and mix
n = 100 # Iteration
num_discrete = 250
cost_fun = lambda x:1000 if x==0 else 1
###################################################

maximize_flag = False
sign = 1


def EI_fun(best_f, mean, std, maximize = True, si = 0.0, cost = None):
    from torch.distributions import Normal

    # deal with batch evaluation and broadcasting
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    u = (mean - best_f.expand_as(mean) + si) / sigma
    
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


bounds = torch.tensor([[-2.0, -2.0, low_b], [2.0, 2.0, high_b]])

samples = draw_sobol_samples(bounds,n= num_discrete,q = 1, batch_shape= None, seed = 12345)
samples = samples.squeeze(1).double()
samples[:,-1] = torch.round(samples[:,-1])

np.random.seed(12345)
high_ind = torch.where(samples[:,-1] == 0)[0]
low_ind = torch.where(samples[:,-1] == 1)[0]
if len(low_ind) > 0:
    Xtrain = torch.vstack([samples[np.random.choice(high_ind, n_high_fidelity), :],samples[np.random.choice(low_ind, n_low_fidelity), :]])
else:
     Xtrain = samples[np.random.choice(high_ind, n_high_fidelity), :]

ytrain = torch.tensor([rosen(Xtrain[i,0:-1].numpy()) 
    if Xtrain[i,-1].numpy() == 0.0 else rosen_low(Xtrain[i,0:-1].numpy()) for i in range(len(Xtrain))])


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
        best_f = ytrain.min().reshape(-1,) 
        gain.append(best_f0 - best_f)
        bestf.append(best_f)

    ######################## Model ####################################    
    model = LMGP(
        train_x=Xtrain,
        train_y=ytrain,
        qual_index= qual_index,
        quant_index= quant_index,
        num_levels_per_var= level_sets,
        quant_correlation_class= quant_kernel,
        NN_layers= [],
        fix_noise= False
    ).double()

    LMGP.reset_parameters

    # optimize noise successively
    reslist,opt_history = fit_model_scipy(
        model, 
        num_restarts = num_minimize_init,
        add_prior=add_prior_flag, # number of restarts in the initial iteration
        n_jobs= 1
    )

    ####################################################################
    
    # Bound is defined as min max of all dimensions. [[min1, min2, min3, ...], [max1, max2, max3, ...]]


    with torch.no_grad():
        ytest, ystd = model.predict(samples, return_std=True)


    print(f'best f is {best_f}')


    cost = torch.tensor(list(map(cost_fun, samples[:,-1])))

    scores = EI_fun(best_f, ytest.reshape(-1,1), ystd.reshape(-1,1), maximize = maximize_flag, cost = cost)

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

    if cumulative_cost[-1] > 99000:
        break

    print(f'Xnew is {Xnew} and ynew is {ynew}')

    if i%10 == 0:

        plt.figure()
        ax = plt.axes(projection = '3d')
        #plt.plot(X,y)
        ax.scatter(Xtrain[:,0], Xtrain[:,1], ytrain, color = 'green')
        ax.scatter(Xnew[0],Xnew[1], ynew, color = 'red')
        

        plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter(samples[:,0], samples[:,1], EI_fun(best_f, ytest.reshape(-1,1), ystd.reshape(-1,1), maximize=maximize_flag).detach())
        ax.scatter(Xnew[0],Xnew[1], scores[index], color = 'red')
        #plt.show()
    
    


plt.figure()
plt.scatter(range(len(ymin_list)), ymin_list)

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