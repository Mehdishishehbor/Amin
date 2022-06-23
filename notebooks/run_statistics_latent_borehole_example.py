import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from statistics_test_Borehole import statistics_test_Borehole

 #####################################################################################
 ###################  Number of data =100        Noise =0
  #####################################################################################
rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

####################################################################################################################################

####################################################################################################################################
out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=100,noise_value=0)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_100_training_noise_0_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_100_training_noise_0_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_100_training_noise_0_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)


#####################################################################################
 ###################  Number of data =50        Noise =0
  #####################################################################################
rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

####################################################################################################################################

####################################################################################################################################
out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=50,noise_value=0)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_50_training_noise_0_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_50_training_noise_0_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_50_training_noise_0_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)




#####################################################################################
 ###################  Number of data =200        Noise =0
  #####################################################################################
rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

####################################################################################################################################

####################################################################################################################################
out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=200,noise_value=0)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_200_training_noise_0_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_200_training_noise_0_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_200_training_noise_0_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)



##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
 #####################################################################################
 ###################  Number of data =50        Noise =1
  #####################################################################################

rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

#rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=50,noise_value=1)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_50_training_noise_1_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_50_training_noise_1_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_50_training_noise_1_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)





##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
 #####################################################################################
 ###################  Number of data =100        Noise =1
  #####################################################################################


rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

#rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=100,noise_value=1)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_100_training_noise_1_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_100_training_noise_1_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_100_training_noise_1_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)





##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################



##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
 #####################################################################################
 ###################  Number of data =200        Noise =1
  #####################################################################################


rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

#rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=200,noise_value=1)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_200_training_noise_1_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_200_training_noise_1_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_200_training_noise_1_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)


##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################

##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
 #####################################################################################
 ###################  Number of data =50        Noise =3
  #####################################################################################

rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

#rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=50,noise_value=3)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_50_training_noise_3_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_50_training_noise_3_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_50_training_noise_3_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)


##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################

##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
 #####################################################################################
 ###################  Number of data =100        Noise =3
  #####################################################################################

rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

#rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=100,noise_value=3)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_100_training_noise_3_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_100_training_noise_3_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_100_training_noise_3_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)

##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################

##################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################

 #####################################################################################
 ###################  Number of data =200        Noise =3
  #####################################################################################


rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_h = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_mse_l3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_opt_history = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_omega_1= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_2= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_3= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_4= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_5= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_6= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_7= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_8= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_9= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_omega_10= {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_a_1 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_2 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_3 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_4 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_5 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_6 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_7 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_8 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_9 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
out_a_10 = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}

out_mse_h_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse_l3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_opt_history_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_omega_1_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_2_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_3_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_4_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_5_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_6_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_7_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_8_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_9_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_omega_10_L= ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']

out_a_1_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_2_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_3_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_4_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_5_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_6_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_7_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_8_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_9_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_a_10_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']


# for l in layers:
#     out_opt_history[str(out_opt_history[l])] = None

# number_of_sources=4
# rep = 5
# for j in range(len(layers)):
#     temp = []
#     for i in range(rep):
#         temp.append(statistics_test(5,rands_seeds * i))
#         #temp.append(statistics_test(rands_seeds * i).numpy().ndarray().tolist())

#     out[str(layers[j])] = temp


temp = []
temp_mse_h = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
# for i in range(rep):
#     MSE_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     MSE
#     for name, mse in zip(layers, MSE_values):
#         out_mse_h[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         MSE_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     MSE
#         for name, mse in zip(layers, MSE_values):
#             out_mse_h[name].append(mse)
    
layers = ['n_init=6']

#rep=20
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]; temp_a_9=[]; temp_a_10=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test_Borehole(randseed=rands_seeds * i,num_train=200,noise_value=3)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])
        temp_a_9.append(positions[4][0]);temp_a_10.append(positions[4][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h

    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8;out_a_9[str(out_a_9_L[j])] = temp_a_9;out_a_10[str(out_a_10_L[j])] = temp_a_10




DATA_borehole_latent_200_training_noise_3_continuation={'out_opt_history':temp_opt_history,'mse':temp_mse_h,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8, 'a_9':temp_a_9,'a_10':temp_a_10,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_borehole_latent_200_training_noise_3_continuation.npy', 'wb') as f:
    np.save(f, DATA_borehole_latent_200_training_noise_3_continuation)

#x=np.load('./DATA_borehole_latent_50_training.npy', allow_pickle=True)