from turtle import forward
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from sklearn.covariance import log_likelihood
import torch
from typing import Any, Optional
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, ConstantDiagLazyTensor
from torch import Tensor

class Multifidelity_likelihood(_GaussianLikelihoodBase):

    def __init__(self, fidel_indices: Tensor , noise_indices: list, noise_prior = None, noise_constraint = None, num_noises = 1,
        learn_additional_noise = False, batch_shape = torch.Size(), **kwargs) -> None:
        
        noise_covar = Multifidelity_noise(noise_prior=noise_prior, 
        noise_constraint=noise_constraint, batch_shape=batch_shape, num_noises = num_noises)
        super().__init__(noise_covar = noise_covar)

        self.fidel_indices = fidel_indices
        self.noise_indices = noise_indices


    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        # This runs the forward method in noise class
        # Shape is not used any more.
        return self.noise_covar(*params, fidel_indices = self.fidel_indices, noise_indices = self.noise_indices)
    

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)





class Multifidelity_noise(_HomoskedasticNoiseBase):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape= torch.Size(), num_noises=1):
        super().__init__(noise_prior, noise_constraint, batch_shape, num_tasks= num_noises)


    def forward(self, *params: Any, shape: Optional[torch.Size] = None, 
        fidel_indices: Tensor, noise_indices: list, **kwargs: Any) -> DiagLazyTensor:
        
        """_summary_
        
        Indices are very important here and is coming from input and shows the multifidelity level of each input data

        Raises:
            ValueError: _description_
        """

        if len(fidel_indices) ==0 or fidel_indices is None:
            raise ValueError('You need to specify a list of indices for noise such as [1,3]')
        # This contains a list of diagonal matrices with defined noise. Crates [batch * noise_size * n * n]
        covar = super().forward(*params, shape= fidel_indices.shape, **kwargs)

        if covar.shape[1] is not len(noise_indices):
            raise ValueError('Something is wrong, number of noise and indices are not the same')

        # This part is for categorical_indices
        temp = ConstantDiagLazyTensor(torch.tensor([0.0]), len(fidel_indices))
        for i in range(len(noise_indices)):
            diag = DiagLazyTensor( (fidel_indices == noise_indices[i]).type(torch.int32) )
            temp += diag * covar[:,i,...]
        
        return temp



if __name__ == '__main__':
    multi_likelihood = Multifidelity_likelihood(num_noises=3)
    multi_noise = Multifidelity_noise(num_noises=2)
    print(multi_likelihood)
    fidel_indices = torch.tensor([1, 3, 2, 3, 2, 1, 3])
    noise_indices = [1, 3]
    covar = multi_noise(fidel_indices = fidel_indices, noise_indices = noise_indices)
    aa = 1