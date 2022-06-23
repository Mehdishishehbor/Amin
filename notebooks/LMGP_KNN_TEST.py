    
def LMGPKNN_test(randseed,Percent_of_missing):  
    
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

    train_x, train_y, test_x, test_y = multifidelity_big_noise['x_train_all'], multifidelity_big_noise['y_train_all'], multifidelity_big_noise['x_val'], multifidelity_big_noise['y_val']

    train_x = torch.from_numpy(np.float64(train_x))
    train_y = torch.from_numpy(np.float64(train_y))
    test_x  = torch.from_numpy(np.float64(test_x))
    test_y  = torch.from_numpy(np.float64(test_y))

    meanx = train_x[:,:-1].mean(dim=-2, keepdim=True)
    stdx = train_x[:,:-1].std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
    train_x[:,:-1] = (train_x[:,:-1] - meanx) / stdx
    test_x[:,:-1] = (test_x[:,:-1] - meanx) / stdx
    train_y = train_y.reshape(-1,)
    test_y = test_y.reshape(-1,)

    ###################################
    Target=train_x[:,-1]
    train_x_withot_lable=np.delete(train_x,10, 1)
    X_Y=torch.concat([train_y.reshape(-1,1),train_x_withot_lable], axis = 1)
    ###################################

    #NAN_Lis=[140,300]
    #Percent_of_missing_data=Percent_of_missing
    Percent_of_missing_data=5
    Number_of_missing_data=np.floor(np.array(train_y.shape)*Percent_of_missing_data/100)
    # NAN_Lis=[1,12,23,44,49,63,73,100,120,148,152,175,189,200,230,260,270,301,318,340]
    random.seed(12)
    #NAN_Lis=random.sample(range(0, int(np.array(train_y.shape))), int(Number_of_missing_data))
    if Percent_of_missing==5:
        NAN_Lis=random.sample(range(0, 49), int(Number_of_missing_data))
    elif Percent_of_missing==10:
        NAN_Lis=random.sample(range(52, 148), int(Number_of_missing_data))
    if Percent_of_missing==20:
        NAN_Lis=random.sample(range(152, 248), int(Number_of_missing_data))
    if Percent_of_missing==30:
        NAN_Lis=random.sample(range(252, 348), int(Number_of_missing_data))
    ###################################
    X_Y_train_input=np.delete(X_Y,NAN_Lis, 0)
    X_Y_train_lable=np.delete(Target,NAN_Lis, 0)

    X_Y_test_input=X_Y[NAN_Lis, :]
    X_Y_test_lable=Target[NAN_Lis]
    ###################################
    neigh = KNeighborsClassifier(n_neighbors=40)
    neigh.fit(np.array(X_Y_train_input.data), np.array(X_Y_train_lable))
    predict=neigh.predict(np.array(X_Y_test_input))

    print('######################################')
    print(f'real_value of missed labels={X_Y_test_lable}')
    print(f'estimated_value with KNeighborsClassifier ={predict}')



    accuracy= np.sum(np.array(X_Y_test_lable) == predict)/len(predict)
    print(f'accuracy rate ={accuracy}')
    print('######################################')
    ####################################################

    # #train_yx_imputation= torch.clone(torch.Tensor([train_y.reshape(-1,1),train_x]))

    # train_yx_imputation=np.copy(np.array([train_y.reshape(-1,1),train_x]))
    # # for k in NAN_Lis:
    # #     train_yx_imputation[k,-1]=np.nan

    # imputer = KNNImputer(n_neighbors=5)
    # train_x_imputted=imputer.fit_transform(train_yx_imputation)

    # estimated_value=[]
    # for k in NAN_Lis:
    #     estimated_value.append(train_x_imputted[k,-1])

    # print(f'estimated_value with KNNImputer ={estimated_value}')
    # print(neigh.predict_proba([[0.9]]))
    ####################################################
    i=5
    for k in NAN_Lis:
        train_x[k,-1]=i
        i+=1
    ###############Parameters########################
    noise_flag = 1
    noise_std = 3.0
    add_prior_flag = True
    num_minimize_init = 24
    qual_index = [10]
    quant_index= list(range(10))
    level_sets =[len(NAN_Lis)+4]# [4]

    ################################## Amin: This part is added to investigate diffirent factors on optimization ########################
    levels_for_predict=[1,2,3,4]
    noise_indices=[] # if you make this [], then you will have one single noise for all cases []
    Optimization_technique='L-BFGS-B' #optimization methods: 'BFGS' 'L-BFGS-B' 'Newton-CG' 'trust-constr'  'SLSQP'
    Optimization_constraint=False   # False # True 
    regularization_parameter=[0, 0] ###### Always at least one element of regularization_parameter should be zero: The first element is for L1 regularization and the second element is for L2
    Bounds=True

    predict_fidelity = 1
    save_mat_flag = False
    quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'
    plt.rcParams['figure.dpi']=150
    plt.rcParams['font.family']='serif'

    #################################################
    # start timing
    start_time = time.time()



    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)



    set_seed(randseed)
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
    #plot_latenth.plot_ls(model2, constraints_flag= True)


    positions=plot_latenth_position.plot_ls(model2, constraints_flag= False)


    ###################################
    neighh = KNeighborsClassifier(n_neighbors=1)
    neighh.fit(np.array(positions[0:4,:].data), np.array([1,2,3,4]))
    predict_based_on_latent_map=neighh.predict(np.array(positions[4:,:].data))

    print('######################################')
    #print(f'real_value of missed labels={X_Y_test_lable}')
    print(f'estimated_value with based_on_latent_map and KNeighborsClassifier= {predict_based_on_latent_map}')

    accuracy= np.sum(np.array(X_Y_test_lable) == predict_based_on_latent_map)/len(predict)
    print(f'accuracy rate of LMGP_KNN ={accuracy}')
    print('######################################')


    ####################################################
    #plt.show()
    return accuracy