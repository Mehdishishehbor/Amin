import torch
import numpy as np

def setlevels(X, qual_index = None):
    if qual_index is None:
        qual_index = list(range(X.shape[-1]))
    if type(X) == torch.Tensor:
        temp = X.clone()
    if X.ndim > 1:
        for j in qual_index:
            a = X[..., j]
            X[..., j] = torch.tensor([*map(lambda m: list(sorted(set(a.tolist()))).index(m), a)])
    else:
        X = torch.tensor([*map(lambda m: list(set(sorted(X.tolist()))).index(m), X)])
    
    if X.dtype == object:
        X = X.astype(float)

    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
        return X
    else:
        return X.to(temp)

