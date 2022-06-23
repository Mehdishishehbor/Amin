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

#!/usr/bin/env python
# coding: utf-8

# # LMGP regresssion demonstration
# 
from pickle import FALSE
import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from lmgp_pytorch.models import LVGPR, LMGP
from lmgp_pytorch.optim import fit_model_scipy,noise_tune
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

from typing import Dict

from lmgp_pytorch.visual import plot_latenth , plot_latenth_position

from lmgp_pytorch.optim import noise_tune
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
#NAN_Lis=[140,300]
Percent_of_missing_data=20
Number_of_missing_data=np.floor(np.array(train_y.shape)*Percent_of_missing_data/100)
# NAN_Lis=[1,12,23,44,49,63,73,100,120,148,152,175,189,200,230,260,270,301,318,340]
random.seed(12)
NAN_Lis=random.sample(range(0, int(np.array(train_y.shape))), 2*int(Number_of_missing_data))

###################################
X_Y_train_input=np.delete(X_Y,NAN_Lis, 0)
X_Y_train_lable=np.delete(Target,NAN_Lis, 0)


#X_Y_test_input=X_Y[NAN_Lis[1:int(Number_of_missing_data)], :]
#X_Y_test_lable=Target[NAN_Lis[1:int(Number_of_missing_data)]]

X_Y_test_input=X_Y[NAN_Lis[int(Number_of_missing_data):], :]
X_Y_test_lable=Target[NAN_Lis[int(Number_of_missing_data):]]
###################################
# neigh = KNeighborsClassifier(n_neighbors=40)
# neigh.fit(np.array(X_Y_train_input.data), np.array(X_Y_train_lable))
# predict=neigh.predict(np.array(X_Y_test_input))

# print('######################################')
# print(f'real_value of missed labels={X_Y_test_lable}')
# print(f'estimated_value with KNeighborsClassifier ={predict}')

# accuracy= np.sum(np.array(X_Y_test_lable) == predict)/len(predict)
# print(f'accuracy rate ={accuracy}')
# print('######################################')
###################################
lissst=[1,3,3,4,55]
accuracy=[]
Number_of_k=100
for k in range(1,Number_of_k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(np.array(X_Y_train_input.data), np.array(X_Y_train_lable))
    predict=neigh.predict(np.array(X_Y_test_input))
    accuracy.append(np.sum(np.array(X_Y_test_lable) == predict)/len(predict))
#print(np.array(accuracy).mean())
print(np.argmax(accuracy))
plt.rcParams.update({'font.size': 19})
fig,axs = plt.subplots(figsize=(8.5,6))
plt.scatter(range(1,Number_of_k),accuracy)
plt.title(f'Data imputation of {Percent_of_missing_data}% of data')
axs.set_xlabel(r'$K$', fontsize = 25)
axs.set_ylabel(r'Accuracy', fontsize = 25)
plt.ylim([.4, .9])
plt.show()
