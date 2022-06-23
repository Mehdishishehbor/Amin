import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25/2
fig = plt.subplots(figsize =(12, 8))
######################################NO_NOISE_NO_Constraint#################################################
# set height of bar

'''
## Obtained MSE
MATLAB = [0.351882399969988, 0.258151299894971, 0.350769645195500, 2.246523451454893]

BFGS = [56.95031085  1.53110033  2.37829401  8.70376905]

LBFGS_B = [1.87597234, 2.17443855, 2.84143082, 9.55008327]

Newton_CG= [2.7223474, 1.57354741, 2.49236741,10.52560847]

trust_constr= [7.33069394,0.43114072,0.7910267,3.05260331]

'''

############################### with_Constraint################################################
# set height of bar



'''
## Estimated noise
MATLAB_Big =[69.541823833284650,9.138499138646997e-04,0.001323962947139,0.187982478341408] 

MATLAB_small = [22.7520,6.460850728457779e-04,6.460850728457779e-04,0.176622784589816]

MATLAB_no = [5.165422521154736e-04,5.165422521154736e-04,5.165422521154736e-04,0.107703612224832]



trust_constr_Big= []

trust_constr_small= [4.0702e+01, 2.4727e-03, 2.3081e-02, 1.6862e-01]

trust_constr_no= [9.0359, 1.5203, 2.1601, 6.1371]

'''

## Obtained MSE
MATLAB_Big  = [1.212971977754596e+02,0.272449346255611,0.347062107472884,2.150866432960034]

MATLAB_small = [41.6794,0.2597,0.3611,2.1883]

MATLAB_no = [0.2931,.2050,0.3119,1.9740]

trust_constr_Big= []

trust_constr_small= [44.7117,  0.0506,  0.0757,  0.6276]

trust_constr_no= [9.0893, 1.5436, 2.4001, 8.7200]

# ########################################################################################

# # Set position of bar on X axis
# br1 = np.arange(len(MATLAB_Big))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
# br4 = [x + barWidth for x in br3]
# br5 = [x + barWidth for x in br4]
# br6 = [x + barWidth for x in br5]

# # Make the plot
# plt.bar(br1, MATLAB_no, color ='r', width = barWidth,
#         edgecolor ='grey', label ='MATLAB no noise')
# plt.bar(br2, trust_constr_no, color ='g', width = barWidth,
#         edgecolor ='grey', label ='trust_constr no noise')
# plt.bar(br3, MATLAB_small, color ='b', width = barWidth,
#         edgecolor ='grey', label ='MATLAB small noise')
# plt.bar(br4, trust_constr_small, color ='c', width = barWidth,
#         edgecolor ='grey', label ='trust_constr small noise')
# plt.bar(br5, MATLAB_Big, color ='y', width = barWidth,
#         edgecolor ='grey', label ='MATLAB big noise')
# plt.bar(br6, trust_constr_Big, color =[1,.5,.5], width = barWidth,
#         edgecolor ='grey', label ='trust_constr big noise')        
      

# # Adding Xticks
# #plt.xlabel('Branch', fontweight ='bold', fontsize = 15)
# plt.ylabel('Estimated noise', fontweight ='bold', fontsize = 15)
# plt.xticks([r + 2.5*barWidth for r in range(len(MATLAB_Big))],
#         ['Based on High fidelity', 'Based on Low fidelity 1', ' Based on Low fidelity 2', ' Based on Low fidelity 3'])

# plt.legend()
# plt.show()
######################################################################################################################################
########################################################################################

# Set position of bar on X axis
br1 = np.arange(len(MATLAB_Big))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]

# Make the plot
plt.bar(br1, MATLAB_no, color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB no noise')
plt.bar(br2, trust_constr_no, color ='g', width = barWidth,
        edgecolor ='grey', label ='trust_constr no noise')
plt.bar(br3, MATLAB_small, color ='b', width = barWidth,
        edgecolor ='grey', label ='MATLAB small noise')
plt.bar(br4, trust_constr_small, color ='c', width = barWidth,
        edgecolor ='grey', label ='trust_constr small noise')
plt.bar(br5, MATLAB_Big, color ='y', width = barWidth,
        edgecolor ='grey', label ='MATLAB big noise')
plt.bar(br6, trust_constr_Big, color =[1,.5,.5], width = barWidth,
        edgecolor ='grey', label ='trust_constr big noise')        
      

# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)
plt.ylabel('Estimated noise', fontweight ='bold', fontsize = 15)
plt.xticks([r + 2.5*barWidth for r in range(len(MATLAB_Big))],
        ['Based on High fidelity', 'Based on Low fidelity 1', ' Based on Low fidelity 2', ' Based on Low fidelity 3'])

plt.legend()
plt.show()