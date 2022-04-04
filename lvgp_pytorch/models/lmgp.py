# Copyright © 2021 by Northwestern University.
# 
# LVGP-PyTorch is copyrighted by Northwestern University. It may be freely used 
# for educational and research purposes by  non-profit institutions and US government 
# agencies only. All other organizations may use LVGP-PyTorch for evaluation purposes 
# only, and any further uses will require prior written approval. This software may 
# not be sold or redistributed without prior written approval. Copies of the software 
# may be made by a user provided that copies are not sold or distributed, and provided 
# that copies are used under the same terms and conditions as agreed to in this 
# paragraph.
# 
# As research software, this code is provided on an "as is'' basis without warranty of 
# any kind, either expressed or implied. The downloading, or executing any part of this 
# software constitutes an implicit agreement to these terms. These terms and conditions 
# are subject to change at any time without prior notice.

import torch
import math
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.priors import NormalPrior,GammaPrior
from gpytorch.distributions import MultivariateNormal
from .gpregression import GPR
from .. import kernels
from ..priors import MollifiedUniformPrior
from ..utils.transforms import softplus,inv_softplus
from typing import List,Optional



class LMGP(GPR):
    """The latent Map GP regression model (LMGP) which extends GPs to handle categorical inputs.

    :note: Binary categorical variables should not be treated as qualitative inputs. There is no 
        benefit from applying a latent variable treatment for such variables. Instead, treat them
        as numerical inputs.

    :param train_x: The training inputs (size N x d). Qualitative inputs needed to be encoded as 
        integers 0,...,L-1 where L is the number of levels. For best performance, scale the 
        numerical variables to the unit hypercube.
    """
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        qual_index:List[int],
        quant_index:List[int],
        num_levels_per_var:List[int],
        lv_dim:int=2,
        quant_correlation_class:str='RBFKernel',
        noise:float=1e-4,
        fix_noise:bool=True,
        lb_noise:float=1e-8,
        NN_layers:list = []
    ) -> None:


        if len(qual_index) > 0:
            qual_kernel = kernels.RBFKernel(
                active_dims=torch.arange(lv_dim)
            )
            qual_kernel.initialize(**{'lengthscale':1.0})
            qual_kernel.raw_lengthscale.requires_grad_(False)


        if len(quant_index) == 0:
            correlation_kernel = qual_kernel
        else:
            try:
                quant_correlation_class = getattr(kernels,quant_correlation_class)
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % quant_correlation_class
                )
            quant_kernel = quant_correlation_class(
                ard_num_dims=len(quant_index),
                active_dims=lv_dim+torch.arange(len(quant_index)),
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log)
            )
            quant_kernel.register_prior(
                'lengthscale_prior',MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
            )
            
            if len(qual_index) > 0:
                correlation_kernel = qual_kernel*quant_kernel
            else:
                correlation_kernel = quant_kernel

        super(LMGP,self).__init__(
            train_x=train_x,train_y=train_y,
            correlation_kernel=correlation_kernel,
            noise=noise,fix_noise=fix_noise,lb_noise=lb_noise
        )

        # register index and transforms
        self.register_buffer('quant_index',torch.tensor(quant_index))
        self.register_buffer('qual_index',torch.tensor(qual_index))


        # latent variable mapping
                # latent variable mapping
        self.num_levels_per_var = num_levels_per_var
        self.lv_dim = lv_dim

        temp = self.transform_categorical(x=torch.tensor(train_x[:,self.qual_index], dtype=torch.int64), type='one-hot')

        # MAPPING
        self.zeta, self.perm = self.zeta_matrix(num_levels=self.num_levels_per_var, lv_dim = self.lv_dim, type='one-hot')
        #nn_model = self.LMMAPPING(num_features = temp.shape[1], type='Linear', lv_dim=2)  
        # Now we add the weigths to the gpytorch module class LMGP 

        model_temp = FFNN(self, input_size=temp.shape[1], num_classes=lv_dim, layers = NN_layers)

        #self.register_parameter('lm', nn_model.weight)
        #self.register_prior(name = 'latent_prior', prior=gpytorch.priors.NormalPrior(0.,1.), param_or_closure='lm')
        
        self.nn_model = model_temp

    def forward(self,x:torch.Tensor) -> MultivariateNormal:

        temp= self.transform_categorical(x=torch.tensor(x[:,self.qual_index], dtype=torch.int64), type='one-hot')

        embeddings = self.nn_model(temp.double())

        if len(self.quant_index) > 0:
            x = torch.cat([embeddings,x[...,self.quant_index]],dim=-1)
        else:
            x = embeddings

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)

    
    def named_hyperparameters(self):
        """Return all hyperparameters other than the latent variables

        This method is useful when different learning rates to the latent variables. To 
        include the latent variables along with others use `.named_parameters` method
        """
        for name, param in self.named_parameters():
            yield name, param
    
    def LMMAPPING(self, num_features:int, type = 'Linear',lv_dim = 2):

        if type == 'Linear':
            in_feature = num_features
            out_feature = lv_dim
            lm = torch.nn.Linear(in_feature, out_feature, bias = False)
            return lm

        else:
            raise ValueError('Only Linear type for now')    

    def zeta_matrix(self,
        num_levels:int,
        lv_dim:int,
        batch_shape=torch.Size(),
        type = 'one-hot'
    ) -> None:

        if any([i == 1 for i in num_levels]):
            raise ValueError('Categorical variable has only one level!')
        elif any([i == 2 for i in num_levels]):
            raise ValueError('Binary categorical variables should not be supplied')

        if lv_dim == 1:
            raise RuntimeWarning('1D latent variables are difficult to optimize!')
        
        for level in num_levels:
            if lv_dim > level - 1:
                lv_dim = min(lv_dim, level-1)
                raise RuntimeWarning(
                    'The LV dimension can atmost be num_levels-1. '
                    'Setting it to %s in place of %s' %(level-1,lv_dim)
                )
    
        if type == 'one-hot':
            from itertools import product
            levels = []
            for l in num_levels:
                levels.append(torch.arange(l))

            perm = list(product(*levels))
            perm = torch.tensor(perm, dtype=torch.int64)
            return self.transform_categorical(perm), perm
        else:
            raise ValueError('Invaid type')

    
    def transform_categorical(self, x:torch.Tensor,
        batch_shape=torch.Size(),
        type = 'one-hot'
        ) -> None:

        if type == 'one-hot':
            # categorical should start from 0
            if x[:,0].min() != 0 or x[:,1].min() != 0:
                x = x -1
            
            x_one_hot = []
            for i in range(x.size()[1]):
                x_one_hot.append( torch.nn.functional.one_hot(x[:,i]) )

            x_one_hot = torch.concat(x_one_hot, axis=1)


            return x_one_hot

from torch import nn
import torch.nn.functional as F 

class FFNN(nn.Module):
    def __init__(self, lmgp, input_size, num_classes, layers):
        super(FFNN, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = nn.Linear(input_size, layers[0]) 
            lmgp.register_parameter('fci', self.fci.weight)
            lmgp.register_prior(name = 'latent_prior_fci', prior=gpytorch.priors.NormalPrior(0.,1.), param_or_closure='fci')

            for i in range(1,self.hidden_num):
                #self.h = nn.Linear(neuran[i-1], neuran[i])
                setattr(self, 'h' + str(i), nn.Linear(layers[i-1], layers[i]))
                lmgp.register_parameter('h'+str(i), getattr(self, 'h' + str(i)).weight )
                lmgp.register_prior(name = 'latent_prior'+str(i), prior=gpytorch.priors.NormalPrior(0.,1.), param_or_closure='h'+str(i))
            
            self.fce = nn.Linear(layers[-1], num_classes)
            lmgp.register_parameter('fce', self.fce.weight)
            lmgp.register_prior(name = 'latent_prior_fce', prior=gpytorch.priors.NormalPrior(0.,1.), param_or_closure='fce')
        else:
            self.fci = nn.Linear(input_size, num_classes, bias = False)
            lmgp.register_parameter('fci', self.fci.weight)
            lmgp.register_prior(name = 'latent_prior_fci', prior=gpytorch.priors.NormalPrior(0.,1.), param_or_closure='fci')


    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        if self.hidden_num > 0:
            x = F.relu(self.fci(x))
            for i in range(1,self.hidden_num):
                #x = F.relu(self.h(x))
                x = F.relu( getattr(self, 'h' + str(i))(x) )
            
            x = self.fce(x)
        else:
            x = self.fci(x)
        return x