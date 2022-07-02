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


import torch
import time
import os
import random
t1 = time.time()

import numpy as np
import matplotlib.pyplot as plt

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")


from botorch.test_functions.multi_fidelity import AugmentedBranin
from lmgp_pytorch.test_functions.multi_fidelity import Augmented_branin
from botorch.utils.sampling import manual_seed

problem = lambda x: Augmented_branin(x, negate=True, mapping = {'0.0': 1.0, '1.0': 0.75, '2.0': 0.5}).to(**tkwargs)
fidelities = torch.tensor([0.0, 1.0, 2.0], **tkwargs)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    manual_seed(seed)


set_seed(1)


# #### Model initialization
# 
# We use a `SingleTaskMultiFidelityGP` as the surrogate model, which uses a kernel from [2] that is well-suited for multi-fidelity applications. The `SingleTaskMultiFidelityGP` models the design and fidelity parameters jointly, so its domain is $[0,1]^7$.

# In[3]:


from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.utils.sampling import draw_sobol_samples

def generate_initial_data(n=16):
    # generate training data
    train_x = draw_sobol_samples(bounds[:,:-1],n=n,q = 1, batch_shape= None).squeeze(1).to(**tkwargs)
    train_f = fidelities[torch.randint(3, (n, 1))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    return train_x_full, train_obj


def initialize_model(train_x, train_obj):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension 6, as in [2]
    model = SingleTaskMultiFidelityGP(train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelity=2)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# #### Define a helper function to construct the MFKG acquisition function
# The helper function illustrates how one can initialize an $q$MFKG acquisition function. In this example, we assume that the affine cost is known. We then use the notion of a `CostAwareUtility` in BoTorch to scalarize the "competing objectives" of information gain and cost. The MFKG acquisition function optimizes the ratio of information gain to cost, which is captured by the `InverseCostWeightedUtility`.
# 
# In order for MFKG to evaluate the information gain, it uses the model to predict the function value at the highest fidelity after conditioning on the observation. This is handled by the `project` argument, which specifies how to transform a tensor `X` to its target fidelity. We use a default helper function called `project_to_target_fidelity` to achieve this.
# 
# An important point to keep in mind: in the case of standard KG, one can ignore the current value and simply optimize the expected maximum posterior mean of the next stage. However, for MFKG, since the goal is optimize information *gain* per cost, it is important to first compute the current value (i.e., maximum of the posterior mean at the target fidelity). To accomplish this, we use a `FixedFeatureAcquisitionFunction` on top of a `PosteriorMean`.

# In[4]:


from botorch import fit_gpytorch_model
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity

from botorch.models.deterministic import DeterministicModel
from torch import Tensor

bounds = torch.tensor([[-5, 0, 0], [10, 15, 1]], **tkwargs)
target_fidelities = {2: 0.0}

class FlexibleFidelityCostModel(DeterministicModel):
    def __init__(
        self,
        fidelity_dims: list = [-1],
         values = {'0.0':1.0, '1.0': 0.50, '2.0': 0.25, '3.0': 0.125},
         fixed_cost: float = 0.01,
         )->None:
        r'Gets the cost according to the fidelity level'
        super().__init__()
        self.cost_values=values
        self.fixed_cost=fixed_cost
        self.fidelity_dims=fidelity_dims
        self.register_buffer("weights", torch.tensor([1.0]))
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        
        cost = list(map(lambda x: self.cost_values[str(float(x))], X[..., self.fidelity_dims].flatten()))
        cost = torch.tensor(cost).to(X)
        cost.reshape(X[..., self.fidelity_dims].shape)
        return self.fixed_cost + cost

#cost_model = AffineFidelityCostModel(fidelity_weights={2: 1.0}, fixed_cost=5.0)
cost_model = FlexibleFidelityCostModel(values = {'0.0':1.0, '1.0': 0.75, '2.0': 0.5, '3.0': 0.25}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_mfkg(model):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[0.0],
    )
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=32 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )
        
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=32 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


# #### Define a helper function that performs the essential BO step
# This helper function optimizes the acquisition function and returns the batch $\{x_1, x_2, \ldots x_q\}$ along with the observed function values. The function `optimize_acqf_mixed` sequentially optimizes the acquisition function over $x$ for each value of the fidelity $s \in \{0, 0.5, 1.0\}$.

# In[5]:


from botorch.optim.optimize import optimize_acqf_mixed


torch.set_printoptions(precision=3, sci_mode=False)

NUM_RESTARTS = 5 if not SMOKE_TEST else 2
RAW_SAMPLES = 32 if not SMOKE_TEST else 4
BATCH_SIZE = 2


def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{2: 0.0}, {2: 1.0}, {2: 2.0}],
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
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


# ### Perform a few steps of multi-fidelity BO
# First, let's generate some initial random data and fit a surrogate model.

from lmgp_pytorch.preprocessing.numericlevels import  setlevels

train_x, train_obj = generate_initial_data(n=16)

#train_x = setlevels(train_x, [2])

cumulative_cost = 0.0
N_ITER = 15 if not SMOKE_TEST else 1

cumulative_cost_hist = []
bestf = []

for i in range(N_ITER):
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_model(mll)
    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    train_x = torch.cat([train_x, new_x])
    #train_x = setlevels(train_x, [2])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost
    cumulative_cost_hist.append(cumulative_cost.clone().cpu().numpy())
    best_f = train_obj.max().reshape(-1,) 
    bestf.append(best_f.clone().cpu().item())



# ### Make a final recommendation
# In multi-fidelity BO, there are usually fewer observations of the function at the target fidelity, so it is important to use a recommendation function that uses the correct fidelity. Here, we maximize the posterior mean with the fidelity dimension fixed to the target fidelity of 1.0.

# In[8]:


def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[0.0],
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


# In[9]:


final_rec = get_recommendation(model)
print(f"\ntotal cost: {cumulative_cost}\n")

print(f'Total time is {time.time() - t1}')

print(cumulative_cost_hist)
print(bestf)


plt.figure()
plt.plot(cumulative_cost_hist, np.array(bestf, dtype = np.float32), 'b.-')
plt.show()



'''
# ### Comparison to standard EI (always use target fidelity)
# Let's now repeat the same steps using a standard EI acquisition function (note that this is not a rigorous comparison as we are only looking at one trial in order to keep computational requirements low).

# In[10]:


from botorch.acquisition import qExpectedImprovement


def get_ei(model, best_f):

    return FixedFeatureAcquisitionFunction(
        acq_function=qExpectedImprovement(model=model, best_f=best_f),
        d=7,
        columns=[6],
        values=[1],
    )


def optimize_ei_and_get_observation(ei_acqf):
    """Optimizes EI and returns a new candidate, observation, and cost."""

    candidates, _ = optimize_acqf(
        acq_function=ei_acqf,
        bounds=bounds[:, :-1],
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # add the fidelity parameter
    candidates = ei_acqf._construct_X_full(candidates)

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


# In[11]:


cumulative_cost = 0.0

train_x, train_obj = generate_initial_data(n=16)

for _ in range(N_ITER):
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_model(mll)
    ei_acqf = get_ei(model, best_f=train_obj.max())
    new_x, new_obj, cost = optimize_ei_and_get_observation(ei_acqf)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    cumulative_cost += cost

final_rec = get_recommendation(model)
print(f"\ntotal cost: {cumulative_cost}\n")


# In[12]:

'''


