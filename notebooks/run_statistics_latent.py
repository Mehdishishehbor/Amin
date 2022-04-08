import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from borehole1D_lmgp2_function import latent_map_NN

rands_seeds = 11245
layers = [[], [10], [10,10], [8,6,4] ]
out = {}
for l in layers:
    out[str(l)] = None


rep = 25
for j in range(len(layers)):
    temp = []
    for i in range(rep):
        temp.append(latent_map_NN(layers[j], rands_seeds * i).numpy().tolist())
    out[str(layers[j])] = temp


data = out.values()
plt.boxplot(data, labels = out.keys())
plt.xlabel('NN Hidden layers')
plt.ylabel('MSE')
plt.show()

with open('out_latent.npy', 'wb') as f:
    np.save(f, out)

with open('out_latent.csv', 'w') as f:
    for key in out.keys():
        f.write("%s,%s\n"%(key,out[key]))

