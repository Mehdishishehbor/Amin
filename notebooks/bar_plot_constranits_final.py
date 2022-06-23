import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25/2
fig = plt.subplots(figsize =(12, 8))





############################### NO_NOISE_NO_Constraint################################################
# set height of bar
'''
## Obtained MSE
MATLAB_WITH_CONS = [0.2931,.2050,0.3119,1.9740]
MATLAB_WITH_OUT_CONS = [0.351882399969988, 0.258151299894971, 0.350769645195500, 2.246523451454893]

MATLAB_WITH_OUT_CONS = [9.948,1.7949,2.7124,10.0746] #without bound
MATLAB_WITH_OUT_CONS = [0.328568850918821,0.215177395234656,0.312679585524057,2.144955746110577] #with bound # interior-point
MATLAB_WITH_OUT_CONS = [0.717598332985581,0.1753,0.5917,1.2193] #without bound # interior-point



L_BFGS_B_with_bound_without_prior= [0.7032, 0.1621, 0.3839, 1.4487]
L_BFGS_B_without_bound_without_prior= [1.5647, 0.1708, 0.2808, 0.9415]

L_BFGS_B_without_bound_with_prior= [0.5803, 0.0663, 0.0971, 0.7536]
L_BFGS_B_with_bound_with_prior= [0.1410, 0.0668, 0.0987, 0.8617]




SLSQP_with_bound_without_prior= [0.1602, 0.0678, 0.0944, 0.8138]
SLSQP_without_bound_without_prior= [1.5647, 0.1708, 0.2808, 0.9415]
SLSQP_without_bound_with_prior= [0.5803, 0.0663, 0.0971, 0.7536]
SLSQP_with_bound_with_prior= []



trust_constr_WITH_CONS= [1.6245, 1.5533, 2.4385, 9.1983]
trust_constr_WITH_CONS= [8.2419, 0.0554, 0.0782, 0.5902]#without proir


trust_constr_WITH_OUT_CONS= [1.7459,  1.6513,  2.6139, 10.2718] #without proir
trust_constr_WITH_OUT_CONS= [13.5202,  0.1086,  0.1776,  1.4802]

SLSQP_with_bound_without_prior= [0.1602, 0.0678, 0.0944, 0.8138]
SLSQP_with_bound_with_regularization_001= [0.1503, 0.0727, 0.0992, 0.8292]
SLSQP_with_bound_with_regularization_01= []


SLSQP_without_bound_without_prior= [1.7429,  1.6489,  2.6143, 10.2445]

SLSQP_without_bound_with_prior= [0.1407, 0.0648, 0.0964, 0.8363]
SLSQP_with_bound_with_prior= [0.1408, 0.0646, 0.0959, 0.8313]

SLSQP_with_bound_with_prior_with_constraint= [0.1408, 0.0646, 0.0959, 0.8312]
SLSQP_without_bound_with_prior_with_constraint= [0.1912, 0.0674, 0.1030, 0.8806]
SLSQP_without_bound_without_prior_with_constraint= [1.8721, 1.8158, 3.0368, 9.8361]
SLSQP_with_bound_without_prior_with_constraint= [0.1504, 0.0728, 0.0992, 0.8284]



SLSQP_without_bound_without_prior_with_regularization_1000= [10621.6235, 12316.5418,  4962.6168, 42803.8993]
SLSQP_without_bound_without_prior_with_regularization_500= [0.1596, 0.0816, 0.0952, 0.9019]
SLSQP_without_bound_without_prior_with_regularization_100= []
SLSQP_without_bound_without_prior_with_regularization_10= []
SLSQP_without_bound_without_prior_with_regularization_1= [1.9042,  1.8635,  2.7735, 10.0692]
SLSQP_without_bound_without_prior_with_regularization_01= []

############################### Estimated noise
MATLAB_WITH_CONS = [5.165422521154736e-04,5.165422521154736e-04,5.165422521154736e-04,0.107703612224832]
MATLAB_WITH_OUT_CONS = [0.000542748775716249,0.000542748775716249,0.00327460764791279,0.138800602039080]
MATLAB_WITH_OUT_CONS = [9.4238,1.5986,2.5419,10.3614] #without bound


trust_constr_WITH_CONS= [1.3126, 1.6317, 2.1652, 6.8674]
trust_constr_WITH_CONS= [0.05971359 0.00492761 0.00468866 0.00754783] #without proir


trust_constr_WITH_OUT_CONS= [1.0531, 1.7042, 2.5040, 8.0007] #without proir
trust_constr_WITH_OUT_CONS= [1.0227e+01, 3.6544e-03, 3.5664e-02, 3.5115e-01]
MATLAB_WITH_OUT_CONS = [0.000663362550024249,0.000636063364238602,0.000628495745141332,0.204163543538034] #with bound # interior-point
MATLAB_WITH_OUT_CONS = [0.004387447927518,0.002448561066115,0.011642897140059,0.172641101296832] #without bound # interior-point


L_BFGS_B_with_bound_without_prior= [4.5748e-04, 4.9260e-05, 1.9674e-02, 3.3425e-01]
L_BFGS_B_without_bound_without_prior= [2.1393e-08, 5.5886e-09, 1.3182e-09, 2.6966e-01]

L_BFGS_B_without_bound_with_prior= [0.0578, 0.0040, 0.0262, 0.1835]
L_BFGS_B_with_bound_with_prior= [0.0330, 0.0030, 0.0246, 0.2026]




SLSQP_with_bound_without_prior= [1.9516e-16, 1.9516e-16, 2.5576e-02, 1.8441e-01]
SLSQP_without_bound_without_prior= [2.1393e-08, 5.5886e-09, 1.3182e-09, 2.6966e-01]
SLSQP_without_bound_with_prior= [0.0578, 0.0040, 0.0262, 0.1835]
SLSQP_with_bound_with_prior= []



SLSQP_with_bound_without_prior= [1.9516e-16, 1.9516e-16, 2.5576e-02, 1.8441e-01]
SLSQP_with_bound_with_regularization_001= [3.1246e-02, 3.3977e-06, 2.5863e-02, 1.9179e-01]
SLSQP_with_bound_with_regularization_01= []

SLSQP_without_bound_without_prior= [1.0577, 1.7065, 2.5114, 7.9974]

SLSQP_without_bound_with_prior= [0.0338, 0.0042, 0.0249, 0.1981]
SLSQP_with_bound_with_prior= [0.0349, 0.0040, 0.0249, 0.1970]

SLSQP_with_bound_with_prior_with_constraint= [0.0349, 0.0040, 0.0249, 0.1970]
SLSQP_without_bound_with_prior_with_constraint= [0.0269, 0.0028, 0.0264, 0.2260]
SLSQP_without_bound_without_prior_with_constraint= [1.3731, 1.7182, 2.7765, 7.5025]
SLSQP_with_bound_without_prior_with_constraint= [3.1428e-02, 1.9516e-16, 2.5837e-02, 1.9104e-01]

SLSQP_without_bound_without_prior_with_regularization_1000= [179.2335, 152.0903,  50.4045,  31.5554]
SLSQP_without_bound_without_prior_with_regularization_500= [2.9345e-02, 1.9520e-16, 2.1341e-02, 2.8879e-01]
SLSQP_without_bound_without_prior_with_regularization_10= []
SLSQP_without_bound_without_prior_with_regularization_1= [1.5160, 1.7671, 2.5597, 8.2828]
SLSQP_without_bound_without_prior_with_regularization_01= []

########################################################################################



############################### Small_NOISE_NO_Constraint################################################
# set height of bar

## Obtained MSE
MATLAB_WITH_CONS = [41.6794,0.2597,0.3611,2.1883]

MATLAB_WITH_OUT_CONS = [44.870647955408984,0.281407718625025,0.375801180059720,2.311783371858700]

trust_constr_WITH_CONS= [44.7117,  0.0506,  0.0757,  0.6276]

trust_constr_WITH_OUT_CONS= [58.0395,  0.1053,  0.1690,  1.2274]


## Estimated noise
MATLAB_WITH_CONS = [22.7520,6.460850728457779e-04,6.460850728457779e-04,0.176622784589816]

MATLAB_WITH_OUT_CONS = [25.120079874285892,6.062377264525291e-04,7.760439013979680e-04,0.228607587872479]

trust_constr_WITH_CONS= [4.0702e+01, 2.4727e-03, 2.3081e-02, 1.6862e-01]

trust_constr_WITH_OUT_CONS= [4.1792e+01, 9.7226e-04, 3.2895e-02, 2.1786e-01]




########################################################################################

######################################BIG_NOISE_BIG_Constraint#################################################
# set height of bar


## Obtained MSE
MATLAB_WITH_CONS = [1.212971977754596e+02,0.272449346255611,0.347062107472884,2.150866432960034]

MATLAB_WITH_OUT_CONS = [1.920262418748112e+02,0.269125036147346,0.356135456844182,2.198408636443548]

trust_constr_WITH_CONS= [1.2090e+02, 1.0433e-01, 1.6547e-01, 1.1591e+00]

trust_constr_WITH_OUT_CONS= [1.5577e+02, 5.7701e-02, 8.3183e-02, 6.3019e-01]


L_BFGS_B_with_bound_without_prior= [4.1486e+02, 5.9678e-02, 5.0606e-02, 3.4437e-01]
L_BFGS_B_without_bound_without_prior= [154.0682,   0.1604,   0.2440,   0.8666]
L_BFGS_B_with_bound_with_prior= [1.2078e+02, 5.5506e-02, 8.2474e-02, 6.8487e-01]
L_BFGS_B_without_bound_with_prior= [1.5636e+02, 5.8194e-02, 8.3569e-02, 6.5732e-01]#without bound



SLSQP_with_bound_without_prior= [1.2074e+02, 5.6722e-02, 8.0157e-02, 6.0056e-01]
SLSQP_without_bound_without_prior= [154.0682,   0.1604,   0.2440,   0.8666]
SLSQP_without_bound_with_prior= [1.5636e+02, 5.8194e-02, 8.3569e-02, 6.5732e-01]
SLSQP_with_bound_with_prior= [1.2078e+02, 5.5506e-02, 8.2474e-02, 6.8487e-01]


SLSQP_with_bound_without_prior= [1.2074e+02, 5.6722e-02, 8.0157e-02, 6.0056e-01]
SLSQP_without_bound_without_prior= [150.9130,   2.3104,   3.3528,  11.0245]

SLSQP_without_bound_with_prior= [1.2078e+02, 5.2604e-02, 7.9391e-02, 6.5808e-01]
SLSQP_with_bound_with_prior= [1.2078e+02, 5.2603e-02, 7.9429e-02, 6.5812e-01]

SLSQP_with_bound_without_prior_with_regularization_10= [1.2080e+02, 5.1812e-02, 7.6690e-02, 6.5479e-01]
SLSQP_with_bound_without_prior_with_regularization_1= [1.2073e+02, 5.5089e-02, 7.8997e-02, 5.9632e-01]
SLSQP_with_bound_without_prior_with_regularization_01= [1.2081e+02, 5.1720e-02, 7.7151e-02, 6.4164e-01]


SLSQP_without_bound_without_prior_with_regularization_1000= [127.5053,   2.3170,   2.8699,  10.1077]
SLSQP_without_bound_without_prior_with_regularization_100= [116.0318,   1.1553,   1.6748,   5.8526]
SLSQP_without_bound_without_prior_with_regularization_10= [121.9953,   1.5286,   2.3478,   8.5822]
SLSQP_without_bound_without_prior_with_regularization_1= [121.9773,   1.5153,   2.3548,   8.6261]
SLSQP_without_bound_without_prior_with_regularization_01= [121.9756,   1.5140,   2.3556,   8.6315]



'''
## Estimated noise
MATLAB_WITH_CONS = [69.541823833284650,9.138499138646997e-04,0.001323962947139,0.187982478341408]

MATLAB_WITH_OUT_CONS = [17.879808654352342,7.006657499408272e-04,6.864906819386295e-04,0.152812069761209]

trust_constr_WITH_CONS= [1.1469e+02, 1.9847e-04, 3.3746e-02, 1.8763e-01]

trust_constr_WITH_OUT_CONS= [1.2568e+02, 1.9570e-04, 2.5666e-02, 1.6098e-01]


L_BFGS_B_with_bound_without_prior= [9.2924e-16, 1.9570e-16, 1.9570e-16, 1.9570e-16]
L_BFGS_B_without_bound_without_prior= [1.2378e+02, 2.0539e-08, 1.3476e-07, 2.3966e-01]


L_BFGS_B_with_bound_with_prior= [1.2024e+02, 1.8235e-03, 2.7634e-02, 2.0128e-01]
L_BFGS_B_without_bound_with_prior= [1.3720e+02, 1.7448e-03, 2.3672e-02, 1.5294e-01]


SLSQP_with_bound_without_prior= [1.1293e+02, 1.9570e-16, 2.4718e-02, 1.4462e-01]
SLSQP_without_bound_without_prior= [154.0682,   0.1604,   0.2440,   0.8666]
SLSQP_without_bound_with_prior= [1.3720e+02, 1.7448e-03, 2.3672e-02, 1.5294e-01]
SLSQP_with_bound_with_prior= [1.2024e+02, 1.8235e-03, 2.7634e-02, 2.0128e-01]


SLSQP_with_bound_without_prior= [1.1293e+02, 1.9570e-16, 2.4718e-02, 1.4462e-01]
SLSQP_without_bound_without_prior=[129.9449,   1.8424,   2.8955,   8.4108] 

SLSQP_without_bound_with_prior= [1.1402e+02, 3.5923e-03, 2.3780e-02, 1.8363e-01]
SLSQP_with_bound_with_prior= [1.1409e+02, 3.5999e-03, 2.3773e-02, 1.8333e-01]


SLSQP_with_bound_without_prior_with_regularization_10= [1.1315e+02, 2.8207e-03, 2.3679e-02, 1.8445e-01]
SLSQP_with_bound_without_prior_with_regularization_1= [1.1154e+02, 7.4790e-15, 2.3480e-02, 1.3849e-01]
SLSQP_with_bound_without_prior_with_regularization_01= [1.1307e+02, 2.5327e-03, 2.3278e-02, 1.7211e-01]

SLSQP_without_bound_without_prior_with_regularization_1000= [114.9977,   1.1508,   1.6658,   6.0843]
SLSQP_without_bound_without_prior_with_regularization_100= [123.0318,   2.2470,   2.8054,   9.7571]
SLSQP_without_bound_without_prior_with_regularization_10= [115.8737,   1.5456,   2.0886,   6.1063]
SLSQP_without_bound_without_prior_with_regularization_1= [115.8296,   1.5387,   2.0954,   6.0922]
SLSQP_without_bound_without_prior_with_regularization_01= [115.8281,   1.5380,   2.0963,   6.0913]
########################################################################################


# Set position of bar on X axis
br1 = np.arange(len(MATLAB_WITH_CONS))
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


# Make the high_fidility plot
plt.bar(br1[1], MATLAB_WITH_CONS[0], color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB with constraint')
plt.bar(br2[1], MATLAB_WITH_OUT_CONS[0], color ='g', width = barWidth,
        edgecolor ='grey', label ='MATLAB ')
plt.bar(br3[1], trust_constr_WITH_CONS[0], color ='b', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint')
plt.bar(br4[1], trust_constr_WITH_OUT_CONS[0], color ='c', width = barWidth,
        edgecolor ='grey', label ='trust_constr ')

      
     
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
plt.bar(br1[0:-1], MATLAB_WITH_CONS[1:], color ='r', width = barWidth,
        edgecolor ='grey', label ='MATLAB with constraint')
plt.bar(br2[0:-1], MATLAB_WITH_OUT_CONS[1:], color ='g', width = barWidth,
        edgecolor ='grey', label ='MATLAB ')
plt.bar(br3[0:-1], trust_constr_WITH_CONS[1:], color ='b', width = barWidth,
        edgecolor ='grey', label ='trust_constr with constraint')
plt.bar(br4[0:-1], trust_constr_WITH_OUT_CONS[1:], color ='c', width = barWidth,
        edgecolor ='grey', label ='trust_constr')

      
     
# Adding Xticks
#plt.xlabel('Branch', fontweight ='bold', fontsize = 15)

plt.ylabel('Estimated noise', fontweight ='bold', fontsize = 15)
plt.xticks([r + 2*barWidth for r in range(len(MATLAB_WITH_CONS[1:]))],
        [ 'Low fidelity 1', 'Low fidelity 2', 'Low fidelity 3'])

##Estimated noise



plt.legend()
plt.show()