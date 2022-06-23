import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from Multi_noise_for_statestics import statistics_test

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

temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[];
temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
temp_a_1=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]
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

rep=50
num_minimize_init=['none','exp','sinh','none']
for j in range(len(layers)):

    temp_mse_h = [];temp_mse_l1 = [];temp_mse_l2 = [];temp_mse_l3 = []

    temp_opt_history = []

    temp_omega_1=[]; temp_omega_2=[]; temp_omega_3=[]; temp_omega_4=[]; temp_omega_5=[]
    temp_omega_6=[]; temp_omega_7=[]; temp_omega_8=[]; temp_omega_9=[]; temp_omega_10=[]
    
    temp_a_1=[];temp_a_2=[]; temp_a_3=[]; temp_a_4=[]; temp_a_5=[]; temp_a_6=[]; temp_a_7=[]; temp_a_8=[]
    
    for i in range(rep):

        #######################################
        MSE_values,omegas,positions,opt_history=statistics_test(randseed=rands_seeds * i)
#       #######################################
        omegas=np.array(omegas[0])

        ######################################

        temp_mse_h.append(MSE_values[0]);temp_mse_l1.append(MSE_values[1]);temp_mse_l2.append(MSE_values[2]);temp_mse_l3.append(MSE_values[3])
        
        temp_opt_history.append(opt_history)

        temp_omega_1.append(omegas[0]);temp_omega_2.append(omegas[1]);temp_omega_3.append(omegas[2]);temp_omega_4.append(omegas[3]);temp_omega_5.append(omegas[4])
        temp_omega_6.append(omegas[5]);temp_omega_7.append(omegas[6]);temp_omega_8.append(omegas[7]);temp_omega_9.append(omegas[8]);temp_omega_10.append(omegas[9])
        
        temp_a_1.append(positions[0][0]);temp_a_2.append(positions[0][1]);temp_a_3.append(positions[1][0]);temp_a_4.append(positions[1][1])
        temp_a_5.append(positions[2][0]);temp_a_6.append(positions[2][1]);temp_a_7.append(positions[3][0]);temp_a_8.append(positions[3][1])


    out_opt_history[str(out_opt_history_L[j])] = temp_opt_history
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h; out_mse_l1[str(out_mse_l1_L[j])] = temp_mse_l1; out_mse_l2[str(out_mse_l2_L[j])] = temp_mse_l2;out_mse_l3[str(out_mse_l3_L[j])] = temp_mse_l3 
    
    out_omega_1[str(out_omega_1_L[j])] = temp_omega_1;out_omega_2[str(out_omega_2_L[j])] = temp_omega_2;out_omega_3[str(out_omega_3_L[j])] = temp_omega_3;out_omega_4[str(out_omega_4_L[j])] = temp_omega_4;out_omega_5[str(out_omega_5_L[j])] = temp_omega_5
    out_omega_6[str(out_omega_6_L[j])] = temp_omega_6;out_omega_7[str(out_omega_7_L[j])] = temp_omega_7;out_omega_8[str(out_omega_8_L[j])] = temp_omega_8;out_omega_9[str(out_omega_9_L[j])] = temp_omega_9;out_omega_10[str(out_omega_10_L[j])] = temp_omega_10

    out_a_1[str(out_a_1_L[j])] = temp_a_1;out_a_2[str(out_a_2_L[j])] = temp_a_2;out_a_3[str(out_a_3_L[j])] = temp_a_3;out_a_4[str(out_a_4_L[j])] = temp_a_4;out_a_5[str(out_a_5_L[j])] = temp_a_5
    out_a_6[str(out_a_6_L[j])] = temp_a_6;out_a_7[str(out_a_7_L[j])] = temp_a_7;out_a_8[str(out_a_8_L[j])] = temp_a_8

'''
temp_a_1.to_csv("submission_FEAT.csv",index=False)

temp_mse_h.to_csv("submission_FEAT.csv",index=False)
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





DATA__constraint={'out_opt_history':temp_opt_history,'mse_h':temp_mse_h, 'mse_l1':temp_mse_l1, 'mse_l2':temp_mse_l2,'mse_l3':temp_mse_l3,'a_1':temp_a_1, 'a_2':temp_a_2, 'a_3':temp_a_3,'a_4':temp_a_4,'a_5':temp_a_5, 'a_6':temp_a_6, 'a_7':temp_a_7,'a_8':temp_a_8,'omega_1':temp_omega_1, 'omega_2':temp_omega_2, 'omega_3':temp_omega_3,'omega_4':temp_omega_4,'omega_5':temp_omega_5, 'omega_6':temp_omega_6, 'omega_7':temp_omega_7,'omega_8':temp_omega_8, 'omega_9':temp_omega_9,'omega_10':temp_omega_10}

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA__constraint.npy', 'wb') as f:
    np.save(f, DATA__constraint)

x=np.load('./DATA__constraint.npy', allow_pickle=True)

# with open('out_latent.csv', 'w') as f:
#     for key in DATA_a_non.keys():
#         f.write("%s,%s\n"%(key,DATA_a_non[key]))
'''
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))
#data = out_mse_h.values()
plt.boxplot(out_mse_h.values(), labels = out_mse_h.keys())
plt.title('MSE high fidelity')
plt.ylabel('MSE')
plt.xlabel('Number of initial points in optimization ')
#plt.show()



plt.figure(figsize=(8, 6))
plt.boxplot(out_mse_l1.values(), labels = out_mse_l1.keys())
plt.title('MSE low fidelity 1')
plt.ylabel('MSE')
plt.xlabel('Number of initial points in optimization ')
#plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(out_mse_l2.values(), labels = out_mse_l2.keys())
plt.title('MSE low fidelity 2')
plt.ylabel('MSE')
plt.xlabel('Number of initial points in optimization ')
#plt.show()


plt.figure(figsize=(8, 6))
plt.boxplot(out_mse_l3.values(), labels = out_mse_l3.keys())
plt.title('MSE low fidelity 3')
plt.ylabel('MSE')  
plt.xlabel('Number of initial points in optimization ')
#plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(out_opt_history.values(), labels = out_opt_history.keys())
plt.title('Objective function')
plt.ylabel('Objective function')
plt.xlabel('Number of initial points in optimization ')  
#plt.show()


plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_1.values(), labels = out_omega_1.keys())
plt.title('omega_1')
#plt.ylabel('MSE')  
plt.xlabel('Number of initial points in optimization ')
#plt.show()


plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_2.values(), labels = out_omega_2.keys())
plt.title('omega_2')
#plt.ylabel('MSE')  
plt.xlabel('Number of initial points in optimization ')
#plt.show()


plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_3.values(), labels = out_omega_3.keys())
plt.title('omega_3')
#plt.ylabel('MSE')  
plt.xlabel('Number of initial points in optimization ')
#plt.show()


plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_4.values(), labels = out_omega_4.keys())
plt.title('omega_4')
#plt.ylabel('_omega_4') 
#  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_5.values(), labels = out_omega_5.keys())
plt.title('omega_5')
#plt.ylabel('_omega_5')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_6.values(), labels = out_omega_6.keys())
plt.title('omega_6')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_7.values(), labels = out_omega_7.keys())
plt.title('omega_7')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_8.values(), labels = out_omega_8.keys())
plt.title('omega_8')
#plt.ylabel('_omega_4')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_9.values(), labels = out_omega_9.keys())
plt.title('omega_9')
#plt.ylabel('_omega_5')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_omega_10.values(), labels = out_omega_10.keys())
plt.title('omega_10')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')





plt.figure(figsize=(8, 6))
plt.boxplot(out_a_1.values(), labels = out_a_1.keys())
plt.title('a_1,1')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_a_2.values(), labels = out_a_2.keys())
plt.title('a_1,2')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_a_3.values(), labels = out_a_3.keys())
plt.title('a_2,1')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_a_4.values(), labels = out_a_4.keys())
plt.title('a_2,2')
#plt.ylabel('_a_4')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_a_5.values(), labels = out_a_5.keys())
plt.title('a_3,1')
#plt.ylabel('_a_5')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_a_6.values(), labels = out_a_6.keys())
plt.title('a_3,2')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_a_7.values(), labels = out_a_7.keys())
plt.title('a_4,1')
#plt.ylabel('MSE')  
#plt.show()
plt.xlabel('Number of initial points in optimization ')

plt.figure(figsize=(8, 6))
plt.boxplot(out_a_8.values(), labels = out_a_8.keys())
plt.title('a_4,2')
#plt.ylabel('_a_4')  
plt.xlabel('Number of initial points in optimization ')
plt.show()

# with open('out_latent.npy', 'wb') as f:
#     np.save(f, out_mse_h)

# with open('out_latent.csv', 'w') as f:
#     for key in out_mse_h.keys():
#         f.write("%s,%s\n"%(key,out_mse_h[key]))

'''