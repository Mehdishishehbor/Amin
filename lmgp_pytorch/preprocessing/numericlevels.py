import torch

def setlevels(X, qual_index):
    if X.ndim > 1:
        for j in qual_index:
            a = X[..., j]
            X[..., j] = torch.tensor([*map(lambda m: list(set(sorted(a.tolist()))).index(m), a)]).to(X)
    else:
        X = torch.tensor([*map(lambda m: list(set(sorted(X.tolist()))).index(m), X)]).to(X)
    return X

