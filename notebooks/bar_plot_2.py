import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25/2
fig = plt.subplots(figsize =(12, 8))
#######################################################################################
# set height of bar
# Matlab_Without_cons = [28, 6, 16, 5, 10]

# Matlab_cons = [12, 30, 1, 8, 22]

# Python_Without_cons = [29, 3, 24, 25, 17]

# Python_cons= [20, 31, 2, 2, 1]

# Methods=[MATLAB,BFGS,LBFGSB,Newton_CG,trust_constr]
# NO_nois=[..., 6, 16, 5, 10]
# Small_nois=[..., 6, 16, 5, 10]
# Large_nois=[..., 6, 16, 5, 10]

Based_on_high_fidelity= [28, 6, 16, 6, 16]

Based_on_low_fidelity = [28, 6, 16, 6, 16]

Based_on_low_fidelity = [12, 30, 1, 6, 16]

Based_on_low_fidelity = [29, 3, 24]


########################################################################################
# Set position of bar on X axis
br1 = np.arange(len(BFGS))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

# Make the plot
plt.bar(br1, MATLAB, color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB')
plt.bar(br2, BFGS, color ='r', width = barWidth,
        edgecolor ='grey', label ='BFGS')
plt.bar(br3, LBFGSB, color ='g', width = barWidth,
        edgecolor ='grey', label ='LBFGSB ')
plt.bar(br4, Newton_CG, color ='b', width = barWidth,
        edgecolor ='grey', label ='Newton_CG')
plt.bar(br5, trust_constr, color ='y', width = barWidth,
        edgecolor ='grey', label ='trust_constr')

methods=[1,2,3,4,5]
# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)
plt.ylabel('MSE', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(0,5)],
        #['High fidelity', 'Low fidelity 1', 'Low fidelity 2', 'Low fidelity 3', 'Low fidelity 4'])
        #['No noise', 'Small noise', 'Large Noise',])
        ['MATLAB', 'BFGS', 'LBFGSB','Newton_CG','trust_constr'])
plt.legend()
plt.show()