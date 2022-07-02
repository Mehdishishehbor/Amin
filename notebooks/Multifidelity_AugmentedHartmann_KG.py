#!/usr/bin/env python
# coding: utf-8

# In[41]:

import numpy as np
import matplotlib.pyplot as plt
import torch

from lmgp_pytorch.models.lmgp import LMGP
from lmgp_pytorch.optim.mll_scipy import fit_model_scipy

from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity

from botorch.test_functions.multi_fidelity import AugmentedHartmann
import os

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
    #"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")


###############Parameters########################
add_prior_flag = False
num_minimize_init = 7
qual_index = [6]
quant_index= [0,1,2,3,4,5]
level_sets = [3]
predict_fidelity = 1

quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'
#################################################

n_high_fidelity = 5
n_low_fidelity = 5
# 1 is high fidelity and 0 is low fidelity. I changed them so that I can define the cost function

# I think this is maximization
problem = AugmentedHartmann(negate=True)
fidelities = torch.tensor([0.5, 0.75, 1.0])


def generate_initial_data(n=16):
    # generate training data
    train_x = torch.rand(n, 6)
    train_f = fidelities[torch.randint(3, (n, 1))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    return train_x_full, train_obj


X_init, Y_init = generate_initial_data()

X = X_init
y = Y_init

maximize_flag = True


Xtrain = X_init

ytrain = Y_init
ytrain = ytrain.reshape(-1,)


import time
import os
t1 = time.time()

ymin_list = []
xmin_list = []

SMOKE_TEST = os.environ.get("SMOKE_TEST")

bounds = torch.tensor([[0.0] * problem.dim, [1.0] * problem.dim])
target_fidelities = {6: 1.0}

cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


def get_mfkg(model):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=7,
        columns=[6],
        values=[1],
    )
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )
        
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


from botorch.optim.optimize import optimize_acqf_mixed

torch.set_printoptions(precision=3, sci_mode=False)

NUM_RESTARTS = 5 if not SMOKE_TEST else 2
RAW_SAMPLES = 128 if not SMOKE_TEST else 4
BATCH_SIZE = 4

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{6: 0.5}, {6: 0.75}, {6: 1.0}],
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()

    new_obj = problem(new_x).unsqueeze(-1)
    
    new_obj = new_obj.reshape(-1,)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost




def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=7,
        columns=[6],
        values=[1],
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    
    final_rec = rec_acqf._construct_X_full(final_rec)
    
    objective_value = problem(final_rec)
    print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
    return final_rec

bestf = []
cumulative_cost_hist = []
cumulative_cost = 0.0
N_ITER = 3 if not SMOKE_TEST else 1

for i in range(N_ITER):

    #_____________________________Best_f________________________________________________________
    if maximize_flag:
        best_f = ytrain.max().reshape(-1,) 
        bestf.append(best_f.clone().item())
    else:
        best_f = ytrain.min().reshape(-1,) 
        bestf.append(best_f.clone().item())

    #__________________________________Initialize model___________________________________________
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

    #LMGP.reset_parameters

    #___________________________________Fitting the model_______________________________________
    reslist,opt_history = fit_model_scipy(
        model, 
        num_restarts = num_minimize_init,
        add_prior=add_prior_flag, # number of restarts in the initial iteration
        n_jobs= 1
    )

    ####################################################################

    print(f'This was the {i+1} iteration')
    #print(f'Batch_candidates = {batch_candidates}')
    #print(f'batch_acq_values = {batch_acq_values}')

    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    Xtrain = torch.cat([Xtrain, new_x])
    ytrain = torch.cat([ytrain, new_obj])
    cumulative_cost += cost
    cumulative_cost_hist.append(cumulative_cost.clone().numpy())

    if i%5 == 0:

        plt.figure()
        ax = plt.axes(projection = '3d')
        #plt.plot(X,y)
        ax.scatter(Xtrain[:,0], Xtrain[:,1], ytrain, color = 'green')
        ax.scatter(new_x[0,0],new_x[0,1], new_obj, color = 'red')
        


final_rec = get_recommendation(model)
print(f"\ntotal cost: {cumulative_cost}\n")


print(f'Total time is {time.time() - t1}')

print(cumulative_cost_hist)
print(bestf)


plt.figure()
plt.plot(cumulative_cost_hist, np.array(bestf, dtype = np.float32), 'b.-')
plt.show()