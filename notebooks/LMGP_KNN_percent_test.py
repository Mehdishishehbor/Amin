    
def LMGP_KNN_percent_test(randseed,Percent_of_missing):  

    from pickle import FALSE
    import torch
    import math
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    import time
    import numpy as np
    from sklearn.impute import KNNImputer
    from sklearn.neighbors import KNeighborsClassifier
    from lmgp_pytorch.utils.input_space import InputSpace
    from lmgp_pytorch.preprocessing.numericlevels import setlevels
    #!/usr/bin/env python
    # coding: utf-8

    # # LMGP regresssion demonstration
    # 
    from pickle import FALSE
    import torch
    import math
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    import time

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    from lmgp_pytorch.models import LVGPR, LMGP
    from lmgp_pytorch.optim import fit_model_scipy,noise_tune
    from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
    from lmgp_pytorch.utils.input_space import InputSpace

    from typing import Dict

    from lmgp_pytorch.visual import plot_latenth , plot_latenth_position

    from lmgp_pytorch.optim import noise_tune
    #----------------------------- Read Data-----------------------

    file_name = './multifidelity_no_noise.mat'
    from scipy.io import loadmat
    multifidelity_big_noise = loadmat(file_name)

    #######################################################################
    num_samples_train = 100
    num_samples_test = 10000
    Percent_of_missing_data=Percent_of_missing

    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        ###############Parameters########################
    noise_flag = 0
    noise_std = 3.0
    add_prior_flag = True
    num_minimize_init = 24
    qual_index = [7]
    quant_index= list(range(7))

    ################################## Amin: This part is added to investigate diffirent factors on optimization ########################
    #levels_for_predict=[1,2,3,4,5]
    levels_for_predict=[1]
    noise_indices=[] # if you make this [], then you will have one single noise for all cases []
    Optimization_technique='L-BFGS-B' #optimization methods: 'BFGS' 'L-BFGS-B' 'Newton-CG' 'trust-constr'  'SLSQP'
    Optimization_constraint=False  # False # True 
    regularization_parameter=[0, 0] ###### Always at least one element of regularization_parameter should be zero: The first element is for L1 regularization and the second element is for L2
    Bounds=True

    predict_fidelity = 1
    save_mat_flag = False
    quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'
    plt.rcParams['figure.dpi']=150
    plt.rcParams['font.family']='serif'

    #######################################################################################
    #######################################################################
    set_seed(200)
    def borehole(params:Dict)->float:
        numerator = 2*math.pi*params['T_u']*(params['H_u']-params['H_l'])
        den_term1 = math.log(params['r']/params['r_w'])
        den_term2 = 1+ 2*params['L']*params['T_u']/(den_term1*params['r_w']**2*params['K_w']) +         params['T_u']/params['T_l']
        
        return numerator/den_term1/den_term2


    # configuration space
    config = InputSpace()
    r = NumericalVariable(name='r',lower=100,upper=50000)
    Tu = NumericalVariable(name='T_u',lower=63070,upper=115600)
    Hu = NumericalVariable(name='H_u',lower=990,upper=1110)
    Tl = NumericalVariable(name='T_l',lower=63.1,upper=116)
    L = NumericalVariable(name='L',lower=1120,upper=1680)
    K_w = NumericalVariable(name='K_w',lower=9855,upper=12045)
    #H_l = NumericalVariable(name='H_l',lower=700,upper=820)
    r_w =NumericalVariable(name='r_w',lower=0.05,upper=0.15)

    #L = CategoricalVariable(name='L',levels=np.linspace(1120,1680,5))
    #K_w = CategoricalVariable(name='K_w', levels=np.linspace(9855,12045,5))
    #r_w = CategoricalVariable(name='r_w',levels=np.linspace(0.05,0.15,5))
    H_l = CategoricalVariable(name='H_l',levels=np.linspace(700,820,5))


    config.add_inputs([r,Tu,Hu,Tl,L,K_w,r_w,H_l])

    ########################################## generate samples ########################################
    train_x = torch.from_numpy(
        config.random_sample(np.random,num_samples_train)
    )
    train_y = [None]*num_samples_train

    for i,x in enumerate(train_x):
        train_y[i] = borehole(config.get_dict_from_array(x.numpy()))

    train_y = torch.tensor(train_y).double()

    if noise_flag == 1:
        train_y += torch.randn(train_y.size()) * noise_std

    # generate 1000 test samples
    num_samples_test = 10000
    test_x = torch.from_numpy(config.random_sample(np.random,num_samples_test))
    test_y = [None]*num_samples_test

    for i,x in enumerate(test_x):
        test_y[i] = borehole(config.get_dict_from_array(x.numpy()))
        
    # create tensor objects
    test_y = torch.tensor(test_y).to(train_y)

    if noise_flag == 1:
        test_y += torch.randn(test_y.size()) * noise_std

    # ## save .mat files
    # if save_mat_flag:
    #     from scipy.io import savemat
    #     savemat('borehole_100_one_CAT_4_Level.mat',{'Xtrain':train_x.numpy(), 'Xtest':test_x.numpy(), 'ytrain':train_y.numpy(), 'ytest':test_y.numpy()})

    ######################################################################################
    #X = setlevels(X, [2])

    train_y=train_y[train_x[:, -1].sort()[1]]
    train_x=train_x[train_x[:, -1].sort()[1]]


    ######################################################################################
    Target=train_x[:,-1]
    train_x_withot_lable=np.delete(train_x,7, 1)
    X_Y=torch.concat([train_y.reshape(-1,1),train_x_withot_lable], axis = 1)
    ###################################

    #NAN_Lis=[140,300]

    Number_of_missing_data=np.floor(np.array(train_y.shape)*Percent_of_missing_data/100)
    # NAN_Lis=[1,12,23,44,49,63,73,100,120,148,152,175,189,200,230,260,270,301,318,340]
    #NAN_Lis=[15,25,65,125,173,220,280,340]
    random.seed(12)
    NAN_Lis=random.sample(range(0, int(np.array(train_y.shape))), int(Number_of_missing_data))

    ###################################
    X_Y_train_input=np.delete(X_Y,NAN_Lis, 0)
    X_Y_train_lable=np.delete(Target,NAN_Lis, 0)

    X_Y_test_input=X_Y[NAN_Lis, :]
    X_Y_test_lable=Target[NAN_Lis]
    ###################################
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(np.array(X_Y_train_input.data), np.array(X_Y_train_lable))
    predict=neigh.predict(np.array(X_Y_test_input))

    print('######################################')
    print(f'real_value of missed labels={X_Y_test_lable}')
    print(f'estimated_value with KNeighborsClassifier ={predict}')



    accuracy= np.sum(np.array(X_Y_test_lable) == predict)/len(predict)
    print(f'accuracy rate ={accuracy}')
    print('######################################')
    ####################################################

    ####################################################
    i=int(torch.max(train_x[:,-1]))+1
    for k in NAN_Lis:
        train_x[k,-1]=i
        i+=1
    ####################################################################################
    ######################################################################################


    # start timing
    start_time = time.time()


    set_seed(randseed)
    level_sets =[int(torch.max(train_x[:, -1]))+1]#[len(NAN_Lis)+5]# [4]
    model2 = LMGP(
        train_x=train_x,
        train_y=train_y,
        noise_indices=noise_indices,
        qual_index= qual_index,
        quant_index= quant_index,
        num_levels_per_var= level_sets,
        quant_correlation_class= quant_kernel,
        NN_layers= [],
        fix_noise= False,
        lb_noise = 1e-20
    ).double()

    LMGP.reset_parameters

    # optimize noise successively
    reslist,opt_history = fit_model_scipy(
        model2, 
        num_restarts = num_minimize_init,
        add_prior=add_prior_flag, # number of restarts in the initial iteration
        n_jobs= 8,
        method = Optimization_technique,
        constraint=Optimization_constraint,
        regularization_parameter=regularization_parameter,
        bounds=Bounds,
    )


    ################################# prediction on test set ########################################
    #################### What fidelity to use for prediction ##################
    test_mean2=[]
    mse=np.zeros(len(levels_for_predict))
    rrmse=np.zeros(len(levels_for_predict))
    j=0
    for i in levels_for_predict:
        index = torch.where(test_x[:,-1] == i)
        test_x_i = test_x[index]
        test_y_i = test_y[index]


        with torch.no_grad():
            # set return_std = False if standard deviation is not needed
            test_mean2 = model2.predict(test_x_i,return_std=False, include_noise = True)
        
        mse[j] = ( (test_y_i-test_mean2)**2).mean().item()
        rrmse[j] = torch.sqrt(((test_y_i-test_mean2)**2).mean()/((test_y_i-test_y_i.mean())**2).mean()).item()
        j+=1
        plt.figure(figsize=(8, 6))
        plt.plot(test_y_i, test_mean2, 'ro')
        plt.plot(test_y_i, test_y_i, 'b')
        plt.xlabel(r'Y_True')
        plt.ylabel(r'Y_predict')
        plt.title(f' Predict based on data source {i}')
        


    # Optimhistory
    print('######################################')

    print(f'finala value of objective function : {opt_history}')


    print('######################################')
    HYPER_Parameter = model2.covar_module.base_kernel.kernels[1].raw_lengthscale.data.numpy().reshape(-1,),
    print(f' HYPER_Parameter are {HYPER_Parameter}')


    print('######################################')
    # print MSE
    print(f'MSE : {torch.tensor(mse)}')


    # print RRMSE
    #rrmse = torch.sqrt(((test_y-test_mean2)**2).mean()/((test_y-test_y.mean())**2).mean())
    print(f'Test RRMSE with noise-tuning strategy : {rrmse}')



    print('######################################')
    noise = model2.likelihood.noise_covar.noise.detach() * model2.y_std**2
    print(f'The estimated noise parameter is {noise}')


    print('######################################')
    positions=plot_latenth_position.plot_ls(model2, constraints_flag= False)
    print(f'The "A" values are {positions}')

    print('######################################')
    # ending timing
    end_time = time.time()
    print(f'The total time in second is {end_time - start_time}')

    # plot latent values
    # plt.figure(figsize=(8, 6))
    plot_latenth.plot_ls(model2, constraints_flag= True)


    positions=plot_latenth_position.plot_ls(model2, constraints_flag= False)


    ###################################
    neighh = KNeighborsClassifier(n_neighbors=1)
    neighh.fit(np.array(positions[0:5,:].data), np.array([0,1,2,3,4]))
    predict_based_on_latent_map=neighh.predict(np.array(positions[5:,:].data))

    print('######################################')
    #print(f'real_value of missed labels={X_Y_test_lable}')
    print(f'estimated_value with based_on_latent_map and KNeighborsClassifier= {predict_based_on_latent_map}')
    accuracy= np.sum(np.array(X_Y_test_lable) == predict_based_on_latent_map)/len(predict)
    print(f'accuracy rate ={accuracy}')
    print('######################################')
    print(f'number of incorrect estimated point ={len(predict)-(np.sum(np.array(X_Y_test_lable) == predict_based_on_latent_map))}')
    print('######################################')

    #plt.show()
    return accuracy