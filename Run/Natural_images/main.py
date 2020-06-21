#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" main module

This module runs the main loops of the network training and testing

"""
import sys
sys.path.append('../../CRBM/')
from crbm import CRBM

import os
import numpy as np

import image_data
import utils
from PIL import Image

# the main entity of the program - an instance of the convolutional DBN
network = None

# will contain the data set to be used by the network.
data = None

model={
       # -- number of bases or "groups" in the layer - equivalent to 
       #       parameter K in the Lee et. al. ICML09 paper
       'num_bases': 24,
       # shape of the bottom-up filter
       'btmup_window_shape': (9, 9),
       # shape of the window used for max pooling
       'block_shape': (2, 2),
                        
       # sparsity parameter - expected activations of the hidden units
       'pbias': 0.008, 
       # step size towards sparsity equilibrium
       'pbias_lambda': 5,  
            
       # initial value for the hiden bias
       'init_bias': 0.01,
       # initial value for visible bias
       'vbias': 0.01,
       # regularization factor to keep the length of weight vector small
       'regL2': 3.5,
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
        'epoch_per_layer': 40,
        'batch':100,
       }

image_params = {
    # path of the directory containing the input images
    'image_path': "../../Dataset/Natural_imgs",    
    # list of image extentions accepted
    'EXTENSIONS':  ['.png', '.jpg', '.tif'],
    # -- maximum edge size in the images - used to scale all the images to have 
    # the same maximum edge size
    'max_image_edge': 512,
    # shape of the patch of the image to be fed to the network
    'image_patch_shape': (10, 10),
    # number of image patches taken from each image before moving to the next
    'samples_per_image': 4000,
    # number of channels in the image (e.g. 1 for balck&white and 3 for RGB)
    'num_channels': 1,
    #the next parameters are for trim the images to suit the max_pooling
    'window_shape': model['btmup_window_shape'],
    'block_shape':model['block_shape'],
    }

# miscellaneous parameters
misc_params = {
    # directory path to save output images, performance plots, etc.
    'results_dir': './results/',
    # name of the file to pickle (save) the network to
    'pickle_fname': 'cdbn_net-natural_imgs.dump',
    # error output file name
    'err_fname': 'error.txt',
    }
# -----------------------------------------------------------------------------
def train():
    
    # -- global variables
    global network
    global data
    
    # -- create a directory to contains the results (oputputs) of the
    # simualtion, if it doesn't exist already
    if not os.path.exists(misc_params['results_dir']):
        os.mkdir(misc_params['results_dir'])
    

    # -------------------------- Read the Input Images ------------------------
    data = image_data.ImageData(image_params)
    data.load_images()
    
    print("Simulation starts with an unlearned network with random weights..\n")
    network = CRBM(model, data.img_shape )
    
    # -- to keep track of error made in each epoch
    err_file = open(misc_params['results_dir']+misc_params['err_fname'], 'w')
    num_epochs = model['epoch_per_layer']

    layer_err = []
   
    for epoch_idx in range(num_epochs):
        print("Training trial #%s.." % epoch_idx)

        torch_batch = data.create_batch()

        for i in range(len(torch_batch)//network.batch):
            print("\n------ Epoch", epoch_idx, ", batch" , i,"------")
            b= torch_batch[i*100:i*100+100,:,:,:]
            print(i,i*100,i*100+100)
            network.contrastive_divergence(b)
                
                
        # -- compute mean of error made in the epoch and save it to the file
        mean_err = np.mean(network.epoch_err)
        layer_err.append(mean_err)
        err_file.write(str(mean_err)+' ')
        err_file.flush()

        # flush the errors made in the previous epoch
        network.epoch_err = []
        
        # -- stop decaying after some point
        # TEST
        if network.std_gaussian > network.model['sigma_stop']:
             network.std_gaussian *= 0.99
                
        # -- visualize layers and save the network at the end of each epoch
        network.visualize_to_files( (2, 12), dir_path=misc_params['results_dir'])
        
        network.save(misc_params['results_dir'] + misc_params['pickle_fname'])
    
    err_file.close()        



if __name__ == "__main__":
    

    train()

