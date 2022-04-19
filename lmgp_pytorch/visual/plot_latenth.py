# LMGP Visualization
# 
#
#

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_ls(model, constraints_flag = True):
    # 
    # plot latent values

    zeta = torch.tensor(model.zeta, dtype = torch.float64)
    #A = model.nn_model.weight.detach()
    perm = model.perm
    levels = model.num_levels_per_var
    #positions = torch.matmul(zeta, A.T)   # this gives the position of each combination in latent space

    positions = model.nn_model(zeta)
    positions = positions.detach()

    # applying the constrains
    if constraints_flag:
        positions = constrains(positions)


    fig = plt.figure(figsize=(8,6))
    colors = {0:'blue', 1:'r', 2:'g', 3:'c', 4:'m', 5:'k', 6:'y'}

    # loop over the number of variables
    for j in range(len(levels)):

        for i in range(levels[j]):
            index = torch.where(perm[:,j] == i) 
            col = list(map(lambda x: colors[x], np.ones(index[0].numpy().shape) * i))
            plt.scatter(positions[index][:,0], positions[index][:,1], label = 'level' + str(i+1), c = col)
            plt.title('Multifidelity', fontsize = 15)
            plt.xlabel(r'$z_1$', fontsize = 15)
            plt.ylabel(r'$z_2$', fontsize = 15)
            plt.legend()
            
        
        fig.tight_layout()
    plt.autoscale()

def constrains(z):
    n = z.shape[0]
    z = z - z[0,:]

    if z[1,0] < 0:
        z[:, 0] *= -1
    
    rot = torch.atan(-1 * z[1,1]/z[1,0])
    R = torch.tensor([ [torch.cos(rot), -1 * torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]])

    z = torch.matmul(R, z.T)
    z = z.T
    if z.shape[1] > 2 and z[2,1] < 0:
        z[:, 1] *= -1
    
    return z