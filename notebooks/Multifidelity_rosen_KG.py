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


from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity


###############Parameters########################
add_prior_flag = False
num_minimize_init = 7
qual_index = [2]
quant_index= [0,1]
level_sets = [2]
predict_fidelity = 1

quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'
#################################################

n_high_fidelity = 5
n_low_fidelity = 5
# 1 is high fidelity and 0 is low fidelity. I changed them so that I can define the cost function

np.random.seed(4)
data = generate_rosen(plot_flag=0)
high = data[np.random.choice(range(0,10000), n_low_fidelity), :]
low = data[np.random.choice(range(10000,20000), n_high_fidelity), :]

train_data = np.vstack([high, low])

X_init = train_data[:,[0,1,3]]

# X_range is [0, 1], therefore we can get the reponse directly  
# from the objective function
# Get the initial responses
Y_init = train_data[:,[2]]


X_init = X_init.astype(np.float64)
Y_init = Y_init.astype(np.float64)

# In[47]:


X = data[:,[0,1,3]]
y = data[:,[2]]

maximize_flag = True
sign = -1

Xtrain = torch.from_numpy(X_init)

ytrain = torch.from_numpy(Y_init)
ytrain = sign * ytrain.reshape(-1,)




import time
import os
t1 = time.time()

ymin_list = []
xmin_list = []

SMOKE_TEST = os.environ.get("SMOKE_TEST")

bounds = torch.tensor([[-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]])
target_fidelities = {2: 1.0}

cost_model = AffineFidelityCostModel(fidelity_weights={2: 999.0}, fixed_cost=1.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


def get_mfkg(model):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
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
BATCH_SIZE = 1

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{2: 0.0}, {2: 1.0}],
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()

    if new_x[0,-1] == 0:
        new_obj = sign * rosen_low([new_x[0, 0].numpy(), new_x[0, 1].numpy()])
    elif new_x[0,-1] == 1:
        new_obj = sign * rosen([new_x[0, 0].numpy(), new_x[0, 1].numpy()])
    
    new_obj = new_obj.reshape(-1,)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost




def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
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
    
    if final_rec[0,-1] == 0:
        objective_value = sign * rosen_low([final_rec[0,0].numpy(), final_rec[0,1].numpy()])
    elif final_rec[0,-1] == 1:
        objective_value = sign * rosen([final_rec[0,0].numpy(), final_rec[0,1].numpy()])

    print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
    return final_rec

bestf = []
cumulative_cost_hist = []
cumulative_cost = 0.0
N_ITER = 5 if not SMOKE_TEST else 1

for i in range(N_ITER):

    #_____________________________Best_f________________________________________________________
    if maximize_flag:
        best_f = ytrain.max().reshape(-1,) 
        bestf.append(best_f.clone().numpy())
    else:
        best_f = ytrain.min().reshape(-1,) 
        bestf.append(best_f.clone().numpy())

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
    Xtrain = torch.cat([Xtrain, torch.tensor(new_x)])
    ytrain = torch.cat([ytrain, torch.tensor(new_obj)])
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
plt.plot(cumulative_cost_hist, np.array(bestf, dtype = np.float32) * -1, 'b.-')
plt.show()