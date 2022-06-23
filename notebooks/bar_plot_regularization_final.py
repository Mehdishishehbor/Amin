import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25/2
fig = plt.subplots(figsize =(12, 8))





############################### NO_NOISE _Constraint################################################
# set height of bar
'''
## Obtained MSE
MATLAB = [0.2931,.2050,0.3119,1.9740]

trust_constr= [9.0893, 1.5436, 2.4001, 8.7200]

trust_constr_Rrg_01= [1.6187, 1.5528, 2.4394, 9.0633]

trust_constr_Rrg_03= [0.2033, 0.0964, 0.1536, 1.3946]

trust_constr_Rrg_05= [1.7357,  1.6387,  2.5911, 10.5660]


## Estimated noise
MATLAB = [5.165422521154736e-04,5.165422521154736e-04,5.165422521154736e-04,0.107703612224832]

trust_constr= [9.0359, 1.5203, 2.1601, 6.1371]

trust_constr_Rrg_01= [1.2409, 1.6144, 2.1356, 6.7278]

trust_constr_Rrg_03= [0.0002, 0.0002, 0.0002, 0.0708]

trust_constr_Rrg_05= [1.1142, 1.6936, 2.3998, 7.1644]
'''
########################################################################################



############################### Small_NOISE_Constraint################################################
# set height of bar
'''
## Obtained MSE
MATLAB= [41.6794,0.2597,0.3611,2.1883]

trust_constr= [44.7117,  0.0506,  0.0757,  0.6276]

trust_constr_Rrg_01= [42.9335,  0.1445,  0.3332,  1.2158]

trust_constr_Rrg_03= [45.7619,  1.5184,  2.4394, 10.3703]

trust_constr_Rrg_05= [45.7614,  1.5185,  2.4393, 10.3682]


## Estimated noise
MATLAB= [22.7520,6.460850728457779e-04,6.460850728457779e-04,0.176622784589816]

trust_constr= [4.0702e+01, 2.4727e-03, 2.3081e-02, 1.6862e-01]

trust_constr_Rrg_01= [4.1010e+01, 2.1022e-04, 1.9583e-02, 2.2135e-01]

trust_constr_Rrg_03= [43.3460,  1.5866,  2.3325,  6.5380]

trust_constr_Rrg_05= [43.3181,  1.5859,  2.3342,  6.5379]



########################################################################################

######################################Big NOISE_Constraint#################################################
# set height of bar


## Obtained MSE
MATLAB = [1.212971977754596e+02,0.272449346255611,0.347062107472884,2.150866432960034]

trust_constr= [1.2090e+02, 1.0433e-01, 1.6547e-01, 1.1591e+00]

trust_constr_Rrg_01= [121.9752,   1.5135,   2.3558,   8.6339]

trust_constr_Rrg_03= [122.0804,   1.5257,   2.4316,  10.3485]

trust_constr_Rrg_05= [122.0791,   1.5257,   2.4317,  10.3486]
'''

## Estimated noise
MATLAB = [69.541823833284650,9.138499138646997e-04,0.001323962947139,0.187982478341408]

trust_constr= [1.1469e+02, 1.9847e-04, 3.3746e-02, 1.8763e-01]

trust_constr_Rrg_01= [115.8232,   1.5374,   2.0961,   6.0899]

trust_constr_Rrg_03= [117.8610,   1.5899,   2.3115,   6.5442]

trust_constr_Rrg_05= [117.9053,   1.5899,   2.3116,   6.5434]

########################################################################################


# Set position of bar on X axis
br1 = np.arange(len(MATLAB))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5= [x + barWidth for x in br4]

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


# Make the high_fidility plot
plt.bar(br1[1], MATLAB[0], color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB')
plt.bar(br2[1], trust_constr[0], color ='g', width = barWidth,
        edgecolor ='grey', label ='trust_constr L=0 ')
plt.bar(br3[1], trust_constr_Rrg_01[0], color ='b', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint L=0.01')
plt.bar(br4[1], trust_constr_Rrg_03[0], color ='c', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint L=0.03 ')
plt.bar(br5[1], trust_constr_Rrg_05[0], color ='y', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint L=0.05 ')

      
     
# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)

plt.ylabel('Estimated noise', fontweight ='bold', fontsize = 15)
plt.xticks([1.25],['High fidelity'])
plt.xlim(0, 2.5) 
##Estimated noise



plt.legend()
plt.show()



#######################################################################################################################
# Make the low_fidility plot
plt.bar(br1[0:-1], MATLAB[1:], color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB')
plt.bar(br2[0:-1], trust_constr[1:], color ='g', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint L=0 ')
plt.bar(br3[0:-1], trust_constr_Rrg_03[1:], color ='b', width = barWidth,
        edgecolor ='grey', label ='trust_constr L=0.01')
plt.bar(br4[0:-1], trust_constr_Rrg_05[1:], color ='c', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint L=0.03')
plt.bar(br5[0:-1], trust_constr_Rrg_05[1:], color ='y', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint L=0.05')

      
     
# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)

plt.ylabel('Estimated noise', fontweight ='bold', fontsize = 15)
plt.xticks([r + 2*barWidth for r in range(len(MATLAB[1:]))],
        [ 'Low fidelity 1', 'Low fidelity 2', 'Low fidelity 3'])

##Estimated noise



plt.legend()
plt.show()