import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25/2
#fig = plt.subplots(figsize =(12, 8))
######################################NO_NOISE_NO_Constraint#################################################
# set height of bar
'''

## Obtained MSE
MATLAB = [0.351882399969988, 0.258151299894971, 0.350769645195500, 2.246523451454893]

BFGS_prior = [0.1408, 0.0646, 0.0959, 0.8313]

LBFGS_B_prior = [0.1407, 0.0652, 0.0980, 0.8498]

Newton_CG_prior= [0.1400, 0.0651, 0.0967, 0.8414]

SLSQP_prior= [0.1407, 0.0648, 0.0964, 0.8363]


MATLAB = [0.351882399969988, 0.258151299894971, 0.350769645195500, 2.246523451454893]

BFGS_without_prior = [2.1231, 0.0879, 0.0860, 0.4996]

LBFGS_B_without_prior= [1.5652, 0.1711, 0.2792, 0.9320]

Newton_CG_without_prior= [1.7845, 0.0721, 0.0968, 0.7262]

SLSQP_without_prior= [1.9671, 1.8770, 2.9707, 9.8595]

#trust_constr= [13.5202,  0.1086,  0.1776,  1.4802]
trust_constr= [1.7459,  1.6513,  2.6139, 10.2718] #without proir


## Estimated noise
MATLAB =[0.000542748775716249,0.000542748775716249,0.00327460764791279,0.138800602039080] 

BFGS_prior = [0.0349, 0.0040, 0.0249, 0.1970]

LBFGS_B_prior = [0.0373, 0.0045, 0.0255, 0.1989]

Newton_CG_prior= [0.0315, 0.0038, 0.0238, 0.1920]

SLSQP_prior= [0.0338, 0.0042, 0.0249, 0.1981]
'''
MATLAB =[0.000542748775716249,0.000542748775716249,0.00327460764791279,0.138800602039080] 

BFGS_without_prior = [2.1613e-16, 1.9622e-16, 1.0906e-11, 8.6280e-02]

LBFGS_B_without_prior = [2.8699e-08, 1.2706e-10, 8.8036e-11, 2.6261e-01]

Newton_CG_without_prior= [2.1393e-08, 5.5886e-09, 1.3182e-09, 2.6966e-01]

SLSQP_without_prior= [0.000578, 0.0040, 0.0262, 0.1835]


#trust_constr= [1.0227e+01, 3.6544e-03, 3.5664e-02, 3.5115e-01]

trust_constr= [1.0531, 1.7042, 2.5040, 8.0007] #without proir

########################################################################################


'''

############################### SMALL_NOISE_NO_Constraint################################################
# set height of bar

## Obtained MSE
MATLAB = [44.870647955408984,0.281407718625025,0.375801180059720,2.311783371858700]

BFGS = [44.4608,  0.1040,  0.1651,  1.1508]

LBFGS_B = [57.6456,  0.1396,  0.2188,  0.8129]

Newton_CG= [44.7769,  0.0948,  0.1491,  1.2878]

Newton_SLSQP= []

trust_constr= [58.0395,  0.1053,  0.1690,  1.2274]

## Estimated noise
MATLAB =[25.120079874285892,6.062377264525291e-04,7.760439013979680e-04,0.228607587872479] 

BFGS = [4.1301e+01, 1.9542e-04, 3.4062e-02, 1.8390e-01]

LBFGS_B = [4.2361e+01, 1.9543e-04, 1.9802e-04, 2.4250e-01]

Newton_CG= [4.3643e+01, 2.2437e-03, 1.8118e-02, 2.0728e-01]

trust_constr= [4.1792e+01, 9.7226e-04, 3.2895e-02, 2.1786e-01]

########################################################################################



############################### Big_NOISE_NO_Constraint################################################
# set height of bar


## Estimated noise
MATLAB =[17.879808654352342,7.006657499408272e-04,6.864906819386295e-04,0.152812069761209] 

BFGS = [1.1409e+02, 3.5988e-03, 2.3773e-02, 1.8334e-01]

LBFGS_B = [1.3720e+02, 1.7448e-03, 2.3672e-02, 1.5294e-01]

Newton_SLSQP= [1.2363e+02, 3.0607e-09, 3.3713e-15, 2.0603e-01]

Newton_CG= [1.1411e+02, 3.6597e-03, 2.2859e-02, 1.7944e-01]

trust_constr= [1.2993e+02, 3.5795e-03, 2.6064e-02, 1.8783e-01]
trust_constr= [1.2628e+02, 3.8378e-05, 1.6635e-02, 2.4523e-01] #without prior

## Obtained MSE
MATLAB = [1.920262418748112e+02,0.269125036147346,0.356135456844182,2.198408636443548]

BFGS = [1.2078e+02, 5.2603e-02, 7.9430e-02, 6.5813e-01]

LBFGS_B = [1.5636e+02, 5.8194e-02, 8.3569e-02, 6.5732e-01]

Newton_CG= [1.2078e+02, 5.2574e-02, 7.9918e-02, 6.5665e-01]

Newton_SLSQP= [1.5752e+02, 1.5072e-01, 3.9603e-01, 1.1954e+00]

trust_constr= [1.5655e+02, 5.4855e-02, 8.4845e-02, 6.8440e-01]

trust_constr= [1.5464e+02, 1.3958e-01, 1.9379e-01, 8.8642e-01] #without prior

'''


########################################################################################

# Set position of bar on X axis
br1 = np.arange(len(MATLAB))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

# # Make the plot
# plt.bar(br1, MATLAB, color ='r', width = barWidth,
#         edgecolor ='grey', label ='MATLAB')
# plt.bar(br2, BFGS, color ='g', width = barWidth,
#         edgecolor ='grey', label ='BFGS')
# plt.bar(br3, LBFGS_B, color ='b', width = barWidth,
#         edgecolor ='grey', label ='L_BFGS_B')
# plt.bar(br4, Newton_CG, color ='c', width = barWidth,
#         edgecolor ='grey', label ='Newton_CG')
# plt.bar(br5, trust_constr, color ='y', width = barWidth,
#         edgecolor ='grey', label ='trust_constr')
      
     
# # Adding Xticks
# #plt.xlabel('Branch', fontweight ='bold', fontsize = 15)

# plt.ylabel('MSE', fontweight ='bold', fontsize = 15)
# plt.xticks([r + 2*barWidth for r in range(len(MATLAB))],
#         ['Based on High fidelity', 'Based on Low fidelity 1', ' Based on Low fidelity 2', ' Based on Low fidelity 3'])

# ##Estimated noise
# plt.ylabel('MSE', fontweight ='bold', fontsize = 15)
# plt.xticks([r + 2*barWidth for r in range(len(MATLAB))],
#         ['High fidelity', ' Low fidelity 1', ' Low fidelity 2', ' Low fidelity 3'])

# plt.legend()
# plt.show()


######################################################################################################################
plt.rcParams.update({'font.size': 19})
fig,axs = plt.subplots(figsize=(9,7))
# Make the low_fidility plot
plt.bar(br1[1], MATLAB[0], color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB')
plt.bar(br2[1], BFGS_without_prior[0], color ='g', width = barWidth,
        edgecolor ='grey', label ='BFGS')
plt.bar(br3[1], LBFGS_B_without_prior[0], color ='b', width = barWidth,
        edgecolor ='grey', label ='L_BFGS_B')
plt.bar(br4[1], Newton_CG_without_prior[0], color ='c', width = barWidth,
        edgecolor ='grey', label ='Newton_CG')
plt.bar(br5[1], SLSQP_without_prior[0], color ='y', width = barWidth,
        edgecolor ='grey', label ='SLSQP')
      
     
# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)

plt.ylabel('Estimated noise', fontweight ='bold')
plt.xticks([1.25],['High fidelity'])
plt.xlim(0, 2.5) 
##Estimated noise



plt.legend()
plt.show()


plt.rcParams.update({'font.size': 19})
fig,axs = plt.subplots(figsize=(9,7))
#######################################################################################################################
# Make the low_fidility plot
plt.bar(br1[0:-1], MATLAB[1:], color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB')
plt.bar(br2[0:-1], BFGS_without_prior[1:], color ='g', width = barWidth,
        edgecolor ='grey', label ='BFGS')
plt.bar(br3[0:-1], LBFGS_B_without_prior[1:], color ='b', width = barWidth,
        edgecolor ='grey', label ='L_BFGS_B')
plt.bar(br4[0:-1], Newton_CG_without_prior[1:], color ='c', width = barWidth,
        edgecolor ='grey', label ='Newton_CG')
plt.bar(br5[0:-1], SLSQP_without_prior[1:], color ='y', width = barWidth,
        edgecolor ='grey', label ='SLSQP')
      
     
# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)

plt.ylabel('Estimated noise', fontweight ='bold')
plt.xticks([r + 2*barWidth for r in range(len(MATLAB[1:]))],
        [ 'Low fidelity 1', 'Low fidelity 2', 'Low fidelity 3'])

##Estimated noise



plt.legend()
plt.show()