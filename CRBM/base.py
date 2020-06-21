#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" base module

This module contains definition of a basis which is a filter (unit) that 
convolves over the output from the previous layer (or the input image for the 
very first layer). Units in the base (represented in 2D arrays of states and
probabilities here) share the same weight matrix and bias, but they receive
different (local) input.

Note: Terms "basis", "base" and "group" are used interchangeably here 
and in the paper

"""

import sys
import time
from PIL import Image
import math
import numpy as np
import scipy as sp
import scipy.signal
#import cPickle
import utils
import config
import torch


# -----------------------------------------------------------------------------
class Base:
    def __init__(self, layer):
        """ Constructor
        
        Input:
            layer -- the layer to which the base belongs to
        
        """
        
        self.layer = layer
        
        self.batch = layer.batch

        # -- copy parameters from layer for the sake of readability
        self.hidden_shape = layer.hidden_shape
        self.num_channels = layer.num_channels
        self.input_shape = layer.input_shape
        self.num_bases = layer.num_bases
        x = torch.tensor(layer.init_bias)
        self.bias = x.repeat( self.num_bases )
        # -- shape of the black of hidden units that compete in max pooling,
        # equivalent to C in the paper
        self.block_shape = self.layer.block_shape
        
        win_shape = layer.btmup_window_shape
        #Xavier weight initialization
        #epsilon_init = 0.01*math.sqrt(6)/math.sqrt(self.num_bases+ (self.input_shape[0]*self.input_shape[1]))
        #self.Wb = torch.rand(self.num_bases, self.num_channels, win_shape[0], win_shape[1]) *2* epsilon_init - epsilon_init

        # Bottom-up weights vector of the group 1 for one minibatch
        self.Wb = 0.001 * torch.randn(self.num_bases, self.num_channels, win_shape[0], win_shape[1])

        # increment in bottom-up weight when updating
        self.Wb_inc = torch.zeros((self.num_bases, self.num_channels, win_shape[0], win_shape[1]))

        # Top-dwon weights vector of the group
        self.Wt = 0  # TODO

        # states of the hidden units during the positive and negative phases
        self.pos_states = torch.zeros((self.num_bases,self.hidden_shape[0],self.hidden_shape[1]))
        self.pos_probs = torch.zeros((self.num_bases,self.hidden_shape[0],self.hidden_shape[1]))
        
        # increment in bias while updating
        self.bias_inc = torch.zeros(self.num_bases)
        
        # output of the pooling units after max pooling operation
        self.pooling_units = torch.zeros((self.batch,self.num_bases,self.layer.output_shape[0],self.layer.output_shape[1]))
        
        #output of activation probabilitie
        self.pooling_probs = torch.zeros((self.batch, self.num_bases,self.layer.output_shape[0],self.layer.output_shape[1]))
        
        # -- activation array the hidden units after positive and negative phases 
        self.pos_activation = 0
        self.neg_activation = 0


        if self.layer.use_cuda:
        # Bottom-up weights vector of the group
            self.Wb = self.Wb.cuda()
            self.Wb_inc = self.Wb_inc.cuda()
            self.pos_states = self.pos_states.cuda() 
            self.pos_probs = self.pos_probs.cuda()
            self.pooling_units = self.pooling_units.cuda()
            self.pooling_probs = self.pooling_probs.cuda()
#-----------------------------------------------------------------------------------------
    
    def sample_prob_max_pooling(self, energy):
        """ Probabilistic Max Pooling
        
        Sample the group, and compute the activation of the hidden and pooling 
        units using the given exponentials of the incoming signals to the units
            
        Input:
            exps -- exponential of the incoming signals (including biases) to 
            the hidden units in the group
        
        Outputs:
            probs -- probability of each unit to become active from its inputs
            states -- binary activation matrix  of the hidden units
            self.pooling_units -- (not a "return" output) - binary activation 
            matrix  of the pooling units
        """       
        # -- create the torch output arrays
        probs = torch.zeros((self.batch,self.num_bases,self.hidden_shape[0],self.hidden_shape[1]))
        states = torch.zeros((self.batch,self.num_bases,self.hidden_shape[0],self.hidden_shape[1]))
        #self.pooling_units[:,:,:]=0
        
        hidden_shape_0 = self.hidden_shape[0]
        hidden_shape_1 = self.hidden_shape[1]

        block_shape_0 = self.block_shape[0]
        block_shape_1 = self.block_shape[1]

        block = torch.zeros((self.batch,self.num_bases,block_shape_0, block_shape_1))
        max_block= torch.empty((self.num_bases,block_shape_0, block_shape_1))
        rnd_val=0
        
        # -- Loop for sampling and max pooling operations
        # Each iteration takes care of one block of units.
        for x_start in range(0, hidden_shape_0, block_shape_0):
            for y_start in range(0, hidden_shape_1, block_shape_1):
                # -- set the start and end indices for the block
                x_end = x_start + block_shape_0
                y_end = y_start + block_shape_1
                
                # block of the units to perform max pooling on
                block = energy[:, :, x_start:x_end, y_start:y_end]
                #print(block.shape)
                max1,_ = block.max(2)
                max_block,_ = max1.max(2)
                #max_block = max_block[:,np.newaxis,np.newaxis]
                #a=block - max_block[:,:,np.newaxis,np.newaxis]
                #print(a[0,0,:,:])
                block_exps= torch.exp(block - max_block[:,:,np.newaxis,np.newaxis])
                #print(probs.shape)
                # whether another unit in the block is active already
                already_chosen = torch.zeros( (self.batch,self.num_bases) ).type(torch.BoolTensor)

                rnd_val = torch.rand( (self.batch,self.num_bases) )

                # TODO: try letting the max index to fire
                # -- compute the state of each unit in the block
                #start = time.time()
                k=0
                for i in range(x_start, x_end):
                    l=0
                    for j in range(y_start, y_end):
                        #  Equation at the end of Section 3.6 in the paper
                        #print(k,l)
                        probs[:,:,i, j] = block_exps[:,:,k, l] / (torch.exp(-1*max_block)+block_exps.sum((3,2)) )
                                                
                        act_index = probs[:,:, i, j] > rnd_val
                        act_index = act_index & ~(already_chosen)
                        vet_temp= states[:,:,i,j]
                        vet_temp[act_index] = 1
                        #print(states[:,:,i,j])
                        already_chosen = act_index | already_chosen
                        
                        l+=1
                    
                    k+=1
                
                #end = time.time()
                #print(end - start)

                        # -- The two lines below are kept only for code readability.
                        #else:  
                        #    states[i, j] = 0
                
                # -- set the activation of the pooling units using their 
                # corresponding hidden units
                i = x_start // block_shape_0  # row of the pooling unit
                j = y_start // block_shape_1  # column of the pooling unit
                #print(already_chosen)
                temp2=self.pooling_units[:,:,i,j]
                temp2[already_chosen]=1
                temp2[~already_chosen]=0
                #print(self.pooling_units[0])
                
                self.pooling_probs[:,:,i,j]= block_exps.sum((3,2)) / (torch.exp(-1*max_block)+block_exps.sum((3,2)))

        #print "self.pooling_units:", np.sum(self.pooling_units)
        return (probs, states)
         
#------------------------------------------------------------------------------------------
    def sample(self, bu_data, td_data=0):
        """ Sampling Given Input Data
        Inputs:
        bu_data -- bottom-up input array
        td_data -- top-down input array
        Outputs:
            probs -- same as probs in function sample_prob_max_pooling()
            states -- same as states in function sample_prob_max_pooling()
            activations -- sum of probabilities to fire (probs) in the group
        """
        
        # -- roughly Equation (3) in the paper
        bu_energy = 0
        
        #timer = utils.Timer('convolution')
        #with timer:  # measures the time
        
        bu_energy += torch.nn.functional.conv2d(bu_data, self.Wb)
        #print(bu_data.shape)
        #print(self.Wb.shape)
        #print(bu_energy.shape)
        
        if config.DEBUG_MODE:        
            assert self.layer.std_gaussian != 0
            
        sigma = 1.0/(self.layer.std_gaussian**2)
        
        bu_energy = sigma * bu_energy + self.bias[np.newaxis,:,np.newaxis,np.newaxis]
        
        # -- roughly Equation (4) in the paper
        td_energy = 0 # TODO: top-down input to a layer 
        energy = bu_energy + td_energy
                    
        #timer = utils.Timer('sample_prob_max_pooling')
        #with timer:  # measures the time
        
        probs, states = self.sample_prob_max_pooling(energy) # P(h|v)
        #print(probs.shape)
        activation = probs.sum((3,2))
        #print(activation.shape)

        return (probs, states, activation)
      
# -----------------------------------------------------------------------------
    def pos_sample(self):
        """ Positive Sampling
        Samples the hidden units during the positive phase of the Gibbs sampling
        """
        
        if config.DEBUG_MODE:
            mystr = "pos_data is nan!"
            assert ~np.isnan(self.layer.pos_data).any(), mystr #'pos_data for %s is nan!' 
            
        #timer = utils.Timer('pos_sample')
        #with timer:  # measures the time
        self.pos_probs, \
        self.pos_states, \
        self.pos_activation = \
        self.sample(self.layer.pos_data)

        # -- debugging assertions
        if config.DEBUG_MODE:
            assert ~np.isnan(self.pos_probs).any()
            assert ~np.isnan(self.pos_states).any()
            assert ~np.isnan(self.pos_activation).any()
  
#------------------------------------------------------------------------------    
    def neg_sample(self):

        
        self.neg_probs, \
        self.neg_states, \
        self.neg_activations = \
            self.sample(self.layer.neg_data)
            
        # -- debugging assertions
        if config.DEBUG_MODE:
            assert ~np.isnan(self.neg_probs).any()
            assert ~np.isnan(self.neg_states).any()
            assert ~np.isnan(self.neg_activations).any()

# -----------------------------------------------------------------------------
    def update(self):
        """ Update Parameters -- the learning component
        Updates the values of the weight vector and biases using the Gradient
        Descent learning rule
        """
        
        # ---------------- (1) Update the weight vector/matrix ----------------
        # number of hidden units
        cnt = self.hidden_shape[0] * self.hidden_shape[1]
        
        # -- compute the convolution of probabilities matrix over data matrix
        # (both positive and negative) added up across channels
        p_prob = torch.zeros(self.num_bases,self.hidden_shape[0],self.hidden_shape[1])
        n_prob = torch.zeros(self.num_bases,self.hidden_shape[0],self.hidden_shape[1])
        """
        #não opt
        for i in range(self.batch):
            p_st= self.pos_states[i,:,:,:]
            p_st= p_st[:,np.newaxis,:,:]
            n_prob= self.neg_probs[i,:,:,:]
            n_prob= n_prob[:,np.newaxis,:,:]
            pos_conv= torch.nn.functional.conv2d(self.layer.pos_data, p_st)
            neg_conv= torch.nn.functional.conv2d(self.layer.neg_data, n_prob )
        """
        p_prob = self.pos_probs.view(1,self.batch*self.num_bases,self.hidden_shape[0],self.hidden_shape[1])
        n_prob = self.neg_probs.view(1,self.batch*self.num_bases,self.hidden_shape[0],self.hidden_shape[1])
        
        
        pos_conv= torch.nn.functional.conv2d(self.layer.pos_data.transpose(0,1), p_prob.transpose(0,1), groups=self.batch)
        neg_conv= torch.nn.functional.conv2d(self.layer.neg_data.transpose(0,1), n_prob.transpose(0,1), groups=self.batch )       
        
        pos_conv=pos_conv.view(self.batch,self.num_bases,self.Wb.shape[2],self.Wb.shape[3])
        neg_conv=neg_conv.view(self.batch,self.num_bases,self.Wb.shape[2],self.Wb.shape[3])
        
        # Gradient descent change (error) in the weight vector
        dW_GD = (pos_conv - neg_conv)/cnt
        dW_GD = dW_GD.sum(0)
        dW_GD = dW_GD[:,np.newaxis,:,:]
        #print(pos_conv.shape)
        #print(dW_GD.shape)
        
        # Regularization change to limit the length of the weight vector
        dW_reg = -1 * self.layer.regL2 * self.Wb 
        #print(dW_reg.shape)
        
        # overall change in the weight vector
        dW = dW_GD  + dW_reg

        # amount of increment in weight vector
        #print(dW.shape)
        #print(dW)
        self.Wb_inc =  self.layer.epsilon/self.batch * dW        
        #print(self.Wb_inc.shape)
        self.Wb += self.Wb_inc
        
        # ---------------- (2) Update the bias of the group -------------------
        # Regularization parameter to enforce sparsity
        sparsity_reg = torch.mean(self.pos_probs,(3,2)) - self.layer.pbias
        sparsity_reg = sparsity_reg.sum(0)
        
        # Gradient descent change (error) in the bias
        dH_GD = (self.pos_activation-self.neg_activations)/cnt 
        dH_GD = dH_GD.sum(0)
        self.bias_inc =  self.layer.epsilon/self.batch * dH_GD - self.layer.pbias_lambda * sparsity_reg/self.batch
        self.bias += self.bias_inc
        #print(self.bias_inc)
