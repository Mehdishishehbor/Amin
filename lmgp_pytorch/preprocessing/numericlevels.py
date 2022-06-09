import torch
import numpy as np

def setlevels(X, qual_index):
    if type(X) == torch.Tensor:
        temp = X.copy()
    if X.ndim > 1:
        for j in qual_index:
            a = X[..., j]
            X[..., j] = torch.tensor([*map(lambda m: list(sorted(set(a.tolist()))).index(m), a)])
    else:
        X = torch.tensor([*map(lambda m: list(set(sorted(X.tolist()))).index(m), X)])
    

    X = X.astype(float)

    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
        return X
    else:
        X.to(temp)

