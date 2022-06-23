import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25/2
#fig = plt.subplots(figsize =(12, 8))
######################################NO_NOISE_NO_Constraint#################################################
# set height of bar
## Obtained MSE
'''
SLSQP_without_cons_without_prior= [1.5647, 0.1708, 0.2808, 0.9415]
SLSQP_without_cons_with_prior= [0.5803, 0.0663, 0.0971, 0.7536]
SLSQP_with_cons_without_prior= [0.2739, 0.0744, 0.1104, 0.9177]
SLSQP_with_cons_with_prior= [0.1407, 0.0648, 0.0964, 0.8363]

'''

'''
### NOISE
SLSQP_without_cons_without_prior= [2.1393e-08, 5.5886e-09, 1.3182e-09, 2.6966e-01]
SLSQP_without_cons_with_prior= [0.0338, 0.0042, 0.0249, 0.1981]
SLSQP_with_cons_without_prior= [0.0251, 0.0020, 0.0285, 0.2550]
SLSQP_with_cons_with_prior= [0.0338, 0.0042, 0.0249, 0.1981]


'''



#############################################big noise #########################
'''

## Obtained MSE
SLSQP_without_cons_without_prior= [154.0682,   0.1604,   0.2440,   0.8666]
SLSQP_without_cons_with_prior= [1.5136e+02, 5.8194e-02, 8.3569e-02, 6.5732e-01]
SLSQP_with_cons_without_prior= [121.9754,   1.5139,   2.3557,   8.6320]
SLSQP_with_cons_with_prior= [1.2079e+02, 5.2701e-02, 7.9387e-02, 6.5931e-01]



'''
### NOISE
SLSQP_without_cons_without_prior= [154.0682,   0.1604,   0.2440,   0.8666]
SLSQP_without_cons_with_prior= [1.3720e+02, 1.7448e-03, 2.3672e-02, 1.5294e-01]
SLSQP_with_cons_without_prior= [115.8286,   1.5379,   2.0964,   6.0911]
SLSQP_with_cons_with_prior= [1.1222e+02, 3.5474e-03, 2.3777e-02, 1.8526e-01]


########################################################################################

# Set position of bar on X axis
br1 = np.arange(len(SLSQP_without_cons_without_prior))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

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
plt.bar(br1[1], SLSQP_without_cons_without_prior[0], color ='r', width = barWidth,
        edgecolor ='grey', label ='without_constraint_without_prior')
plt.bar(br2[1], SLSQP_without_cons_with_prior[0], color ='g', width = barWidth,
        edgecolor ='grey', label ='without_constraint_with_prior')
plt.bar(br3[1], SLSQP_with_cons_without_prior[0], color ='b', width = barWidth,
        edgecolor ='grey', label ='with_constraint_without_prior')
plt.bar(br4[1], SLSQP_with_cons_with_prior[0], color ='c', width = barWidth,
        edgecolor ='grey', label ='with_constraint_with_prior')

      
     
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
plt.bar(br1[0:-1], SLSQP_without_cons_without_prior[1:], color ='r', width = barWidth,
        edgecolor ='grey', label ='without_constraint_without_prior')
plt.bar(br2[0:-1], SLSQP_without_cons_with_prior[1:], color ='g', width = barWidth,
        edgecolor ='grey', label ='without_constraint_with_prior')
plt.bar(br3[0:-1], SLSQP_with_cons_without_prior[1:], color ='b', width = barWidth,
        edgecolor ='grey', label ='with_constraint_without_prior')
plt.bar(br4[0:-1], SLSQP_with_cons_with_prior[1:], color ='c', width = barWidth,
        edgecolor ='grey', label ='with_constraint_with_prior')




# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)

plt.ylabel('Estimated noise', fontweight ='bold')
plt.xticks([r + 2*barWidth for r in range(len(SLSQP_with_cons_with_prior[1:]))],
        [ 'Low fidelity 1', 'Low fidelity 2', 'Low fidelity 3'])

##Estimated noise



plt.legend()
plt.show()