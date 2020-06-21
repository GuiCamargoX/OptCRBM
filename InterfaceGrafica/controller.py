
import sys
sys.path.append('../CRBM/')
from crbm import CRBM

import os
import numpy as np
import utils
from loadDataset import loadDataset
from PyQt5 import QtWidgets, QtGui

class ConnectCRBM:
    global network
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
    
    
    def __init__(self, hyperameters, dataset):
        self.model = hyperameters
        self.dataset = dataset
        
    def train(self, progress, progress_bar, pos_img, neg_img, weights_img, pooling_img):        
        # -- create a directory to contains the results (oputputs) of the
        # simualtion, if it doesn't exist already
        if not os.path.exists(self.misc_params['results_dir']):
            os.mkdir(self.misc_params['results_dir'])
        
        # -------------------------- Read the Input Images ------------------------
        #Dataset
        X_train, X_valid, X_test = loadDataset.read(self.dataset)
        
        #--------------------------------------------------------------------------
        torch_batch = utils.process_array_to_pytorch(X_train, self.model['btmup_window_shape'], self.model['block_shape'] )    
        torch_batch_test = utils.process_array_to_pytorch(X_test, self.model['btmup_window_shape'], self.model['block_shape'] )
    
        
        print("Simulation starts with an unlearned network with random weights..\n")
        self.network = CRBM(self.model, (torch_batch.shape[2],torch_batch.shape[3], torch_batch.shape[1] ) )
    
        # -- to keep track of error made in each epoch
        err_file = open(self.misc_params['results_dir']+self.misc_params['err_fname_training'], 'w')
        err_file2 = open(self.misc_params['results_dir']+self.misc_params['err_fname_test'], 'w')
        num_epochs = self.model['epoch_per_layer']
    
        progress_bar.setMaximum(num_epochs)
        for epoch_idx in range(num_epochs):
            print("Training trial #%s.." % epoch_idx)
    
    
            progress.append("\n\n\nTraining error")
            for i in range(len(torch_batch)//self.network.batch):
                print("\n------ Epoch ", epoch_idx, ", batch " , i,"------")
                b= torch_batch[i*self.model['batch']:i*self.model['batch']+self.model['batch'],:,:,:]
                log = self.network.contrastive_divergence(b)
                progress.append("\n------ Epoch "+ str(epoch_idx) + ", batch "+ str(i)+ "------")
                progress.append(log)
                QtWidgets.QApplication.processEvents()
                    
                    
            # -- compute mean of error made in the epoch and save it to the file
            mean_err = np.mean(self.network.epoch_err)
            err_file.write(str(mean_err)+' ')
            err_file.flush()
    
            # flush the errors made in the previous epoch
            self.network.epoch_err = []
            
            print("\n\n\nTest error")
            progress.append("\n\n\nTest error")
            for i in range(len(torch_batch_test)//self.network.batch):
                print("\n------ Epoch ", epoch_idx, ", batch " , i,"------")
                b= torch_batch_test[i*self.model['batch']:i*self.model['batch']+self.model['batch'],:,:,:]
                print(i,i*self.model['batch'],i*self.model['batch']+self.model['batch'])
                log= self.network.gibbs(b)
                progress.append("\n------ Epoch "+ str(epoch_idx) + ", batch "+ str(i)+ "------")
                progress.append(log)
                QtWidgets.QApplication.processEvents()
            
            # -- compute mean of error made in the epoch and save it to the file
            mean_err = np.mean(self.network.epoch_err)
            err_file2.write(str(mean_err)+' ')
            err_file2.flush()
    
            # flush the errors made in the previous epoch
            self.network.epoch_err = []
                    
            
            # -- stop decaying after some point
            # TEST
            if self.network.std_gaussian > self.network.model['sigma_stop']:
                 self.network.std_gaussian *= 0.99
                    
            # -- visualize layers and save the self.network at the end of each epoch
            self.network.visualize_to_files( (5, 5), dir_path=self.misc_params['results_dir'])
            pos_img.setPixmap(QtGui.QPixmap("results/pos_data.jpg"))
            neg_img.setPixmap(QtGui.QPixmap("results/neg_data.jpg"))
            weights_img.setPixmap(QtGui.QPixmap("results/weights.jpg"))
            pooling_img.setPixmap(QtGui.QPixmap("results/-poooling.jpg"))
            progress_bar.setValue(epoch_idx)
            QtWidgets.QApplication.processEvents()
            
            self.network.save(self.misc_params['results_dir'] + self.misc_params['pickle_fname'])
            
            #imprimir imagem
        progress_bar.setValue(num_epochs)
        err_file.close()
        err_file2.close()    