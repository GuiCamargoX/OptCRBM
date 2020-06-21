#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" main module

This module runs the main loops of the network training and testing

"""

import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../../CRBM/')
from crbm import CRBM

import utils

# the main entity of the program - an instance of the convolutional DBN
network = None

# will contain the data set to be used by the network.
data = None

model={
       # -- number of bases or "groups" in the layer - equivalent to 
       #       parameter K in the Lee et. al. ICML09 paper 5
       'num_bases': 25,
       # shape of the bottom-up filter 10,10
       'btmup_window_shape': (11,11),
       # shape of the window used for max pooling
       'block_shape': (2, 2),
                        
       # sparsity parameter - expected activations of the hidden units
       'pbias': 0.05, 
       # step size towards sparsity equilibrium
       'pbias_lambda': 5,  
            
       # initial value for the hiden bias
       'init_bias': 0.01,
       # initial value for visible bias
       'vbias': 0.001,
       # regularization factor to keep the length of weight vector small
       'regL2': 0.01,
       # learning rate - the ratio of change taken from gradient descent
       'epsilon': 0.1,
       # start and stop value of the parameter used to control the effect 
       # of input vector (versus bias)
        'sigma_start': 0.2,
        'sigma_stop': 0.1,
        # -- number of steps (loops) performed in the contrastive
        # divergence algorithm
        'CD_steps': 1,
        # -- number of epochs to learn each layer after all previous layers are
        # learned
        'epoch_per_layer': 20,
        'batch':100,
       }

# miscellaneous parameters
misc_params = {
    # directory path to save output images, performance plots, etc.
    'results_dir': './results/',
    # name of the file to pickle (save) the network to
    'pickle_fname': 'cdbn_net-natural_imgs.dump',
    # error output file name training
    'err_fname_training': 'error_training.txt',
    # error output file name test
    'err_fname_test': 'error_test.txt',
    }
# -----------------------------------------------------------------------------
def train():
    
    # -- global variables
    global network
    
    # -- create a directory to contains the results (oputputs) of the
    # simualtion, if it doesn't exist already
    if not os.path.exists(misc_params['results_dir']):
        os.mkdir(misc_params['results_dir'])
    

    # -------------------------- Read the Input Images ------------------------
    mnist = tf.keras.datasets.mnist
    (X, Y),(X_test, Y_test) = mnist.load_data()
    
    #from sklearn.model_selection import train_test_split
    #x, X, y, Y = train_test_split(X, Y, stratify =Y, test_size=0.10, random_state=42)
    X_train = X.astype(np.float)/255
    X_test = X_test.astype(np.float)/255
    #Y = Y_test
    
    '''
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize
    
    #X_norm = StandardScaler().fit_transform(X.reshape(X.shape[0],28*28))
    #X_test_norm = StandardScaler().fit_transform(X_test.reshape(X_test.shape[0],28*28))
    X_norm = X.reshape(X.shape[0],28*28)/255
    X_test_norm = X_test.reshape(X_test.shape[0],28*28)/255
    
    # initializing the pca
    pca = PCA(n_components = 361, whiten=True, random_state=42)

    X = pca.fit_transform(X_norm)
    X_t = pca.transform(X_test_norm)
    
    X_train = X.reshape(X.shape[0],19,19)
    X_test = X_t.reshape(X_t.shape[0],19,19)
    '''
    
    #--------------------------------------------------------------------------
    torch_batch = utils.process_array_to_pytorch(X_train, model['btmup_window_shape'], model['block_shape'] )    
    torch_batch_test = utils.process_array_to_pytorch(X_test, model['btmup_window_shape'], model['block_shape'] )

    
    print("Simulation starts with an unlearned network with random weights..\n")
    network = CRBM(model, (torch_batch.shape[2],torch_batch.shape[3], torch_batch.shape[1] ) )
    
    # -- to keep track of error made in each epoch
    err_file = open(misc_params['results_dir']+misc_params['err_fname_training'], 'w')
    err_file2 = open(misc_params['results_dir']+misc_params['err_fname_test'], 'w')
    num_epochs = model['epoch_per_layer']


    for epoch_idx in range(num_epochs):
        print("Training trial #%s.." % epoch_idx)


        for i in range(len(torch_batch)//network.batch):
            print("\n------ Epoch", epoch_idx, ", batch" , i,"------")
            b= torch_batch[i*model['batch']:i*model['batch']+model['batch'],:,:,:]
            print(i,i*model['batch'],i*model['batch']+model['batch'])
            network.contrastive_divergence(b)
                
                
        # -- compute mean of error made in the epoch and save it to the file
        mean_err = np.mean(network.epoch_err)
        err_file.write(str(mean_err)+' ')
        err_file.flush()

        # flush the errors made in the previous epoch
        network.epoch_err = []
        
        print("\n\n\nERROR Test")
        for i in range(len(torch_batch_test)//network.batch):
            print("\n------ Epoch", epoch_idx, ", batch" , i,"------")
            b= torch_batch_test[i*model['batch']:i*model['batch']+model['batch'],:,:,:]
            print(i,i*model['batch'],i*model['batch']+model['batch'])
            network.gibbs(b)
        
        # -- compute mean of error made in the epoch and save it to the file
        mean_err = np.mean(network.epoch_err)
        err_file2.write(str(mean_err)+' ')
        err_file2.flush()

        # flush the errors made in the previous epoch
        network.epoch_err = []
                
        
        # -- stop decaying after some point
        # TEST
        if network.std_gaussian > network.model['sigma_stop']:
             network.std_gaussian *= 0.99
                
        # -- visualize layers and save the network at the end of each epoch
        network.visualize_to_files( (5, 5), dir_path=misc_params['results_dir'])
        
        network.save(misc_params['results_dir'] + misc_params['pickle_fname'])
    
    err_file.close()
    err_file2.close()
    
#------------------------------------------------------------------------------------------

if __name__ == "__main__":
    

    train()

