from pickle import FALSE
import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
#----------------------------- Read Data-----------------------

file_name = './multifidelity_no_noise.mat'
from scipy.io import loadmat
multifidelity_big_noise = loadmat(file_name)

train_x, train_y, test_x, test_y = multifidelity_big_noise['x_train_all'], multifidelity_big_noise['y_train_all'], multifidelity_big_noise['x_val'], multifidelity_big_noise['y_val']

train_x = torch.from_numpy(np.float64(train_x))
train_y = torch.from_numpy(np.float64(train_y))
test_x  = torch.from_numpy(np.float64(test_x))
test_y  = torch.from_numpy(np.float64(test_y))

meanx = train_x[:,:-1].mean(dim=-2, keepdim=True)
stdx = train_x[:,:-1].std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x[:,:-1] = (train_x[:,:-1] - meanx) / stdx
test_x[:,:-1] = (test_x[:,:-1] - meanx) / stdx
train_y = train_y.reshape(-1,)
test_y = test_y.reshape(-1,)

###################################
Target=train_x[:,-1]
train_x_withot_lable=np.delete(train_x,10, 1)
X_Y=torch.concat([train_y.reshape(-1,1),train_x_withot_lable], axis = 1)
###################################

NAN_Lis=[1,12,23,44,49,63,73,100,120,148,152,175,189,200,230,260,270,301,318,340]

###################################
X_Y_train_input=np.delete(X_Y,NAN_Lis, 0)
X_Y_train_lable=np.delete(Target,NAN_Lis, 0)

X_Y_test_input=X_Y[NAN_Lis, :]
X_Y_test_lable=Target[NAN_Lis]
###################################
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(np.array(X_Y_train_input.data), np.array(X_Y_train_lable))
predict=neigh.predict(np.array(X_Y_test_input))
print(f'real_value of missed labels={X_Y_test_lable}')
print(f'estimated_value with KNeighborsClassifier ={predict}')

i=5
for k in NAN_Lis:
    train_x[k,-1]=i
    i+=1




# train_xx= torch.clone(train_x)
# realvalue=[]
# for k in NAN_Lis:
#     realvalue.append(train_xx[k,-1].data.tolist())

# imputer = KNNImputer(n_neighbors=2)
# train_x_imputted=imputer.fit_transform(train_X_Y_3)

# estimated_value=[]
# for k in NAN_Lis:
#     estimated_value.append(train_x_imputted[k,-1])

# print(f'estimated_value with KNNImputer ={estimated_value}')
#print(neigh.predict_proba([[0.9]]))