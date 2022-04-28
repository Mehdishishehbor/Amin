from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
import torch

class Multifidelity_noise(FixedNoiseGaussianLikelihood):

    def __init__(self, noise: Tensor, learn_additional_noise: Optional[bool] = False, batch_shape: Optional[torch.Size] = ..., **kwargs: Any) -> None:
        super().__init__(noise, learn_additional_noise, batch_shape, **kwargs)


