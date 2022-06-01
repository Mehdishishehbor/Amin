from lmgp_pytorch.preprocessing.numericlevels import setlevels
from lmgp_pytorch.preprocessing.Normalize import standard
import torch


############### test1 ############################
torch.manual_seed(12345)

a = torch.tensor([[0.5, 0.75, 0.1, 1.0, 0.5, 0.75, 0.125, 0.1, 1.0, 0.1, 0.5], [0.75, 0.1, 0.5, 0.5, 0.5, 0.75, 0.125, 0.1, 1.0, 0.1, 0.5]])
X = torch.randn([11, 4]) * 6 + 2
X = torch.concat([X, a.T], axis = 1)

X = setlevels(X, [4,5])

X = standard(X, [0,1,2,3])

print(X, '\n')
print(X.mean(axis = 0), '\n')
############### test2 ############################
torch.manual_seed(12345)

a = torch.tensor([[1, 10, 20, 1.0, 20.0, 3.0, 1.0, 5, 7.0, 4, 2], [0.75, 0.1, 0.5, 0.5, 0.5, 0.75, 0.125, 0.1, 1.0, 0.1, 0.5]])
X = torch.randn([11, 4]) * 6 + 2
X = torch.concat([X, a.T], axis = 1)

X = setlevels(X, [4,5])

X = standard(X, [0,1,2,3])

print(X)
print(X.mean(axis = 0), '\n')
############### test3 ############################
torch.manual_seed(12345)

a = torch.tensor([[1, 10, 20, 1.0, 20.0, 3.0, 1.0, 5, 7.0, 4, 2], [0.75, 0.1, 0.5, 0.5, 0.5, 0.75, 0.125, 0.1, 1.0, 0.1, 0.5]])
X = torch.randn([11, 4]) * 6 + 2
X = torch.concat([X, a.T], axis = 1)

X = setlevels(X, [4,5])

X, X2 = standard(X, [0,1,2,3], X*2)

print(X, '\n')
print(X.mean(axis = 0), '\n')
print(X2, '\n')
print(X2.mean(axis = 0), '\n')