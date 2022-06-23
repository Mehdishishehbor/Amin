import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from Multi_noise_for_statestics import statistics_test

rands_seeds = 11245
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
out_mse = {'n_init=6':[], 'n_init=12':[], 'n_init=24':[],'n_init=32':[]}
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

####################################################################################################################################

####################################################################################################################################
out_mse_L = ['n_init=6', 'n_init=12', 'n_init=24','n_init=32']
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
temp_mse = [];temp_mse_l1 = [] ;temp_mse_l2 = [];temp_mse_l3 = []

temp_opt_history = []

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[];
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]
# for i in range(rep):
#     mse_values,omegas,positions=statistics_test(rands_seeds * i)
#     #     mse
#     for name, mse in zip(layers, mse_values):
#         out_mse[name].append(mse)
    
# for j in range(len(num_minimize_init)):
#     for i in range(rep):
#         mse_values,omegas,positions=statistics_test(num_minimize_init[j],rands_seeds * i)
#         #     mse
#         for name, mse in zip(layers, mse_values):
#             out_mse[name].append(mse)
    
layers = ['n_init=6']

# rep=50
# num_minimize_init=['none','exp','sinh','none']
# for j in range(len(layers)):

#     temp_mse = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

#     temp_opt_history = []

#     temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
#     temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
#     temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]
    
#     for i in range(rep):

#         #######################################
#         mse_values,omegas,positions,opt_history=statistics_test(randseed=rands_seeds * i)
# #       #######################################
#         omegas=np.array(omegas[0])

#         ######################################

        # temp_mse.append(mse_values[0]);temp_mse_l1.append(mse_values[1]);temp_mse_l2.append(mse_values[2]);temp_mse_l3.append(mse_values[3])
        
        # temp_opt_history.append(opt_history)

        # temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        # temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6]);temp_omega_8.append(omegas[7]);temp_omega_9.append(omegas[8]);temp_omega_10.append(omegas[9])
        
        # temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        # temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])


    # out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    # out_mse[str(out_mse_L[j])] = temp_mse; out_mse_l1[str(out_mse_l1_L[j])] = temp_mse_l1; out_mse_l2[str(out_mse_l2_L[j])] = temp_mse_l2;out_mse_l3[str(out_mse_l3_L[j])] = temp_mse_l3 
    
    # out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    # out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    # out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    # out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8

'''
temp_a_1.to_csv("submission_FEAT.csv",index=False)

temp_mse.to_csv("submission_FEAT.csv",index=False)
temp_mse_l1.to_csv("submission_FEAT.csv",index=False)
temp_mse_l2.to_csv("submission_FEAT.csv",index=False)
temp_mse_l3.to_csv("submission_FEAT.csv",index=False)

temp_omega_1.to_csv("submission_FEAT.csv",index=False)
temp_omega_2.to_csv("submission_FEAT.csv",index=False)
temp_omega_3.to_csv("submission_FEAT.csv",index=False)
temp_omega_4.to_csv("submission_FEAT.csv",index=False)
temp_omega_5.to_csv("submission_FEAT.csv",index=False)
temp_omega_6.to_csv("submission_FEAT.csv",index=False)
temp_omega_7.to_csv("submission_FEAT.csv",index=False)
temp_omega_8.to_csv("submission_FEAT.csv",index=False)
temp_omega_9.to_csv("submission_FEAT.csv",index=False)
temp_omega_10.to_csv("submission_FEAT.csv",index=False)


temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])

'''





#DATA_exp2={'out_opt_history':temp_opt_history,'mse':temp_mse, 'mse_l1':temp_mse_l1, 'mse_l2':temp_mse_l2,'mse_l3':temp_mse_l3,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6, 'omega_7':temp_omega_7,'omega_8':temp_omega_8, 'omega_9':temp_omega_9,'omega_10':temp_omega_10}

#DATA_one.to_csv("DATA_one",index=False)
# with open('DATA_exp2.npy', 'wb') as f:
#     np.save(f, DATA_exp2)

x=np.load('D:\Pytone\LMGP\lmgp-pmacs-NN_latent\lmgp-pmacs-Multiple_noise_estimate\lmgp-pmacs-Multiple_noise_estimate/DATA_borehole_latent_50_training_noise_0_continuation.npy', allow_pickle=True)
x_exp=np.load('D:\Pytone\LMGP\lmgp-pmacs-NN_latent\lmgp-pmacs-Multiple_noise_estimate\lmgp-pmacs-Multiple_noise_estimate/DATA_borehole_latent_100_training_noise_0_continuation.npy', allow_pickle=True)
x_sinh=np.load('D:\Pytone\LMGP\lmgp-pmacs-NN_latent\lmgp-pmacs-Multiple_noise_estimate\lmgp-pmacs-Multiple_noise_estimate/DATA_borehole_latent_200_training_noise_0_continuation.npy', allow_pickle=True)
# with open('out_latent.csv', 'w') as f:
#     for key in DATA_a_non.keys():
#         f.write("%s,%s\n"%(key,DATA_a_non[key]))

out_mse = {'50 data':x.tolist()['mse'], '100 data':x_exp.tolist()['mse'],'200 data':x_sinh.tolist()['mse']}

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))
#data = out_mse.values()
plt.boxplot(out_mse.values(), labels = out_mse.keys())
plt.ylabel('MSE')
#plt.xlabel('Prior on latent "A" parameters ')
#plt.show()
plt.title(' Continuation No Noise')



out_opt_history = {'50 data':x.tolist()['out_opt_history'][-1], '100 data':x_exp.tolist()['out_opt_history'][-1],'200 data':x_sinh.tolist()['out_opt_history'][-1]}
plt.figure(figsize=(8, 6))
plt.boxplot(out_opt_history.values(), labels = out_opt_history.keys())
plt.ylabel('Objective function')
plt.ylabel('Objective function')
#plt.xlabel('Prior on latent "A" parameters ')  
#plt.show()
plt.title(' Continuation No Noise')


out_omega_1 = {'50 data':x.tolist()['omega_1'], '100 data':x_exp.tolist()['omega_1'],'200 data':x_sinh.tolist()['omega_1']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_1.values(), labels = out_omega_1.keys())
plt.ylabel('omega_1')
#plt.ylabel('mse')  
#plt.xlabel('Prior on latent "A" parameters ')
#plt.show()
plt.title(' Continuation No Noise')

out_omega_2 = {'50 data':x.tolist()['omega_2'], '100 data':x_exp.tolist()['omega_2'],'200 data':x_sinh.tolist()['omega_2']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_2.values(), labels = out_omega_2.keys())
plt.ylabel('omega_2')
#plt.ylabel('mse')  
#plt.xlabel('Prior on latent "A" parameters ')
#plt.show()
plt.title(' Continuation No Noise')

out_omega_3 = {'50 data':x.tolist()['omega_3'], '100 data':x_exp.tolist()['omega_3'],'200 data':x_sinh.tolist()['omega_3']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_3.values(), labels = out_omega_3.keys())
plt.ylabel('omega_3')
#plt.ylabel('mse')  
#plt.xlabel('Prior on latent "A" parameters ')
#plt.show()
plt.title(' Continuation No Noise')

out_omega_4 = {'50 data':x.tolist()['omega_4'], '100 data':x_exp.tolist()['omega_4'],'200 data':x_sinh.tolist()['omega_4']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_4.values(), labels = out_omega_4.keys())
plt.ylabel('omega_4')
#plt.ylabel('_omega_4') 
#  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_omega_5 = {'50 data':x.tolist()['omega_5'], '100 data':x_exp.tolist()['omega_5'],'200 data':x_sinh.tolist()['omega_5']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_5.values(), labels = out_omega_5.keys())
plt.ylabel('omega_5')
#plt.ylabel('_omega_5')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_omega_6 = {'50 data':x.tolist()['omega_6'], '100 data':x_exp.tolist()['omega_6'],'200 data':x_sinh.tolist()['omega_6']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_6.values(), labels = out_omega_6.keys())
plt.ylabel('omega_6')
#plt.ylabel('mse')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')



out_a_1 = {'50 data':x.tolist()['a_1'], '100 data':x_exp.tolist()['a_1'],'200 data':x_sinh.tolist()['a_1']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_1.values(), labels = out_a_1.keys())
plt.ylabel('z_1,1')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_2 = {'50 data':x.tolist()['a_2'], '100 data':x_exp.tolist()['a_2'],'200 data':x_sinh.tolist()['a_2']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_2.values(), labels = out_a_2.keys())
plt.ylabel('z_1,2')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_3 = {'50 data':x.tolist()['a_3'], '100 data':x_exp.tolist()['a_3'],'200 data':x_sinh.tolist()['a_3']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_3.values(), labels = out_a_3.keys())
plt.ylabel('z_2,1')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_4 = {'50 data':x.tolist()['a_4'], '100 data':x_exp.tolist()['a_4'],'200 data':x_sinh.tolist()['a_4']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_4.values(), labels = out_a_4.keys())
plt.ylabel('z_2,2')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_5 = {'50 data':x.tolist()['a_5'], '100 data':x_exp.tolist()['a_5'],'200 data':x_sinh.tolist()['a_5']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_5.values(), labels = out_a_5.keys())
plt.ylabel('z_3,1')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_6 = {'50 data':x.tolist()['a_6'], '100 data':x_exp.tolist()['a_6'],'200 data':x_sinh.tolist()['a_6']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_6.values(), labels = out_a_6.keys())
plt.ylabel('z_3,2')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_7 = {'50 data':x.tolist()['a_7'], '100 data':x_exp.tolist()['a_7'],'200 data':x_sinh.tolist()['a_7']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_7.values(), labels = out_a_7.keys())
plt.ylabel('z_4,1')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_8 = {'50 data':x.tolist()['a_8'], '100 data':x_exp.tolist()['a_8'],'200 data':x_sinh.tolist()['a_8']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_8.values(), labels = out_a_8.keys())
plt.ylabel('z_4,2')  
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')


out_a_9 = {'50 data':x.tolist()['a_9'], '100 data':x_exp.tolist()['a_9'],'200 data':x_sinh.tolist()['a_9']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_9.values(), labels = out_a_9.keys())
plt.ylabel('z_5,1')  
#plt.show()
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')

out_a_10 = {'50 data':x.tolist()['a_10'], '100 data':x_exp.tolist()['a_10'],'200 data':x_sinh.tolist()['a_10']}
plt.figure(figsize=(8, 6))
plt.boxplot(out_a_10.values(), labels = out_a_10.keys())
plt.ylabel('z_5,2')  
#plt.xlabel('Prior on latent "A" parameters ')
plt.title(' Continuation No Noise')
plt.show()