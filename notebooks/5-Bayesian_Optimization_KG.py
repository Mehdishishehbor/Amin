#!/usr/bin/env python
# coding: utf-8

from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.multi_fidelity import multi_fidelity_wing
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.utils import set_seed
from lmgp_pytorch.optim import fit_model_scipy
from lmgp_pytorch.bayesian_optimizations.bo_steps import run_bo_kg
import torch
from botorch.utils.sampling import draw_sobol_samples
from lmgp_pytorch.test_functions.multi_fidelity import Augmented_branin
from lmgp_pytorch.bayesian_optimizations.cost_model import FlexibleFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
###############Parameters########################
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
        }

set_seed(1)

bounds = torch.tensor([[-5, 0, 0], [10, 15, 2]], **tkwargs)
target_fidelities = {2: 0.0}
problem = lambda x: Augmented_branin(x, negate=True, mapping = {'0.0': 1.0, '1.0': 0.75, '2.0': 0.5}).to(**tkwargs)
fidelities = torch.tensor([0.0, 1.0, 2.0], **tkwargs)
################### Initialize Data ###############################
def generate_initial_data(n=16):
# generate training data
    train_x = draw_sobol_samples(bounds[:,:-1],n=n,q = 1, batch_shape= None).squeeze(1).to(**tkwargs)
    train_f = fidelities[torch.randint(3, (n, 1))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    return train_x_full, train_obj

Xtrain_x, train_obj = generate_initial_data(n=16)
train_obj = train_obj.reshape(-1)

######################################################################
qual_index = {2:3}
#model = LMGP(Xtrain_x, train_obj, qual_ind_lev=qual_index)
cost_model = FlexibleFidelityCostModel(values = {'0.0':1.0, '1.0': 0.75, '2.0': 0.5, '3.0': 0.25}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

run_bo_kg(model_name = 'LMGP', train_x = Xtrain_x, train_obj = train_obj, problem = problem, cost_model = cost_model, 
N_ITER = 10, bounds= bounds, target_fidelities= target_fidelities, qual_index = qual_index)