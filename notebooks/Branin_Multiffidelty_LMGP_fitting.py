#!/usr/bin/env python3
# coding: utf-8

# ## Multi-Fidelity BO with Discrete Fidelities using KG
# 
# In this tutorial, we show how to do multi-fidelity BO with discrete fidelities based on [1], where each fidelity is a different "information source." This tutorial uses the same setup as the [continuous multi-fidelity BO tutorial](https://botorch.org/tutorials/multi_fidelity_bo), except with discrete fidelity parameters that are interpreted as multiple information sources.
# 
# We use a GP model with a single task that models the design and fidelity parameters jointly. In some cases, where there is not a natural ordering in the fidelity space, it may be more appropriate to use a multi-task model (with, say, an ICM kernel). We will provide a tutorial once this functionality is in place.
# 
# [1] [M. Poloczek, J. Wang, P.I. Frazier. Multi-Information Source Optimization. NeurIPS, 2017](https://papers.nips.cc/paper/2017/file/df1f1d20ee86704251795841e6a9405a-Paper.pdf)
# 
# [2] [J. Wu, S. Toscano-Palmerin, P.I. Frazier, A.G. Wilson. Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning. Conference on Uncertainty in Artificial Intelligence (UAI), 2019](https://arxiv.org/pdf/1903.04703.pdf)

# ### Set dtype and device

# In[1]:


from keyword import kwlist
import torch
import time
import os
import random
t1 = time.time()

import numpy as np
import matplotlib.pyplot as plt

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")


# ### Problem setup
# 
# We'll consider the Augmented Hartmann multi-fidelity synthetic test problem. This function is a version of the Hartmann6 test function with an additional dimension representing the fidelity parameter; details are in [2]. The function takes the form $f(x,s)$ where $x \in [0,1]^6$ and $s \in \{0.5, 0.75, 1\}$. The target fidelity is 1.0, which means that our goal is to solve $\max_x f(x,1.0)$ by making use of cheaper evaluations $f(x,s)$ for $s \in \{0.5, 0.75\}$. In this example, we'll assume that the cost function takes the form $5.0 + s$, illustrating a situation where the fixed cost is $5.0$.

# In[2]:


from botorch.test_functions.multi_fidelity import AugmentedBranin




# I think this is maximization
problem = AugmentedBranin(negate=False).to(**tkwargs)
fidelities = torch.tensor([0.5, 0.75, 1.0], **tkwargs)


# #### Model initialization
# 
# We use a `SingleTaskMultiFidelityGP` as the surrogate model, which uses a kernel from [2] that is well-suited for multi-fidelity applications. The `SingleTaskMultiFidelityGP` models the design and fidelity parameters jointly, so its domain is $[0,1]^7$.

# In[3]:


from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from lmgp_pytorch.models.lmgp import LMGP
from lmgp_pytorch.optim.mll_scipy import fit_model_scipy

from botorch.utils.sampling import draw_sobol_samples

bounds = torch.tensor([[-5, 0], [10, 15]], **tkwargs)

def generate_initial_data(n=16):
    # generate training data
     
    train_x = draw_sobol_samples(bounds,n=n,q = 1, batch_shape= None).squeeze(1).to(**tkwargs)
    #train_x = torch.rand(n, 6, **tkwargs)
    train_f = fidelities[torch.randint(3, (n, 1))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    return train_x_full, train_obj

###############Parameters########################
add_prior_flag = True
num_minimize_init = 10
qual_index = [2]
quant_index= [0,1]
level_sets = [3]
predict_fidelity = 2

quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

#################################################

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(4)

X, y = generate_initial_data(n=10000)
X[:,-1] = torch.tensor(list(map(lambda x:torch.where(X[:,-1].unique() == x)[0].item(), X[:,-1])))
y = y.reshape(-1)

train_x = X[:50,:]
train_obj = y[:50]

testx = X[1000:10000,:]
testy = y[1000:10000]

index = torch.where(testx[:,-1] == predict_fidelity)
test_x = testx[index]
test_obj = testy[index]



meanx = train_x[:,:-1].mean(dim=-2, keepdim=True)
stdx = train_x[:,:-1].std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x[:,:-1] = (train_x[:,:-1] - meanx) / stdx
test_x[:,:-1] = (test_x[:,:-1] - meanx) / stdx

 #__________________________________Initialize model___________________________________________
model = LMGP(
    train_x=train_x.to(**tkwargs),
    train_y=train_obj.to(**tkwargs),
    qual_index= qual_index,
    quant_index= quant_index,
    num_levels_per_var= level_sets,
    quant_correlation_class= quant_kernel,
    NN_layers= [],
    fix_noise= False
).to(**tkwargs)

#___________________________________Fitting the model_______________________________________
reslist,opt_history = fit_model_scipy(
    model, 
    num_restarts = num_minimize_init,
    add_prior=add_prior_flag, # number of restarts in the initial iteration
    n_jobs= 8
)


with torch.no_grad():
    # set return_std = False if standard deviation is not needed
    test_mean2 = model.predict(test_x.to(**tkwargs),return_std=False, include_noise = True)


for j in range(1):

    mse = ( (test_obj-test_mean2)**2).mean().item()
    rrmse= torch.sqrt(((test_obj-test_mean2)**2).mean()/((test_obj-test_obj.mean())**2).mean()).item()
    plt.figure(figsize=(8, 6))
    plt.plot(test_obj.cpu(), test_mean2.cpu(), 'ro')
    plt.plot(test_obj.cpu(), test_obj.cpu(), 'b')
    plt.xlabel(r'Y_True')
    plt.ylabel(r'Y_predict')
    plt.title(f' Predict based on data source {j}')

plt.show()
print(f'MSE is {mse}')