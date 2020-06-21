import torch
import utils
from base import Base
import numpy as np
from PIL import Image
import pickle

class CRBM():

    def __init__(self, model, input_data_shape ,use_cuda=False):
        self.model = model
        self.use_cuda = use_cuda

        # -- copy values from model to class data members, to improve code
        # readability
        self.block_shape = model['block_shape']               # pooling shape
        self.btmup_window_shape = model['btmup_window_shape'] # convolution window shape
        print("Nw = ", self.btmup_window_shape)
        self.epsilon = model['epsilon']
        self.regL2 = model['regL2']
        self.pbias = model['pbias']
        self.pbias_lambda = model['pbias_lambda']
        self.init_bias = model['init_bias']
        self.vbias = model['vbias']
        self.std_gaussian = model['sigma_start']
        self.CD_steps = model['CD_steps']
        self.num_bases= model['num_bases']
        self.input_shape = input_data_shape
        self.num_channels = input_data_shape[2]
        self.batch= model['batch']
 
        # amount of change made in the bias of the visible layer while updating
        self.vbias_inc = 0
        
        # a list to hold the errors at each training cycle of the epoch
        self.epoch_err = []        
        
        # -- shape of hidden units in each base of the layer
        h = self.input_shape[0] - self.btmup_window_shape[0] + 1
        w = self.input_shape[1] - self.btmup_window_shape[1] + 1
        self.hidden_shape = (h, w)

        # -- shape of output (pooling) units in each base
        h = self.hidden_shape[0] // self.block_shape[0]
        w = self.hidden_shape[0] // self.block_shape[1]
        self.output_shape = (h,w)

        # negative data, i.e. network's belief
        self.neg_data = torch.zeros((self.batch, self.num_channels, self.input_shape[0], self.input_shape[1]))

        # positive data - input from previus layer (raw input if first layer)
        self.pos_data = torch.zeros((self.batch,self.num_channels, self.input_shape[0], self.input_shape[1]))

        # -- create the bases
        self.bases = Base(self)
        #print(self.input_shape)

#-----------------------------------------------------------------------------------------------------
        
    def gibbs(self, visible_nodes):
        """ Update Layer
        
        Performs Gibbs sampling of the layer's state variables (given the
        previous layer), and then updates the parameters and weights
        accordingly. Here is the steps performed:
            1) Sample each base given 
        
        Input:
            layer_to_learn -- index of the layer to be learned
        
        """
        
        self.pos_data= visible_nodes
        print(self.pos_data.shape)
        log=''
        # ------------------------- Prositive Phase --------------------------
        print("\nPositive phase ...")
        log+="Positive phase ...\n"
        
        #timer = utils.Timer('positive phase')
        #with timer:  # measures the time
        #print "self.pos_data:", self.pos_data
        self.bases.pos_sample()
           
        # ------------------------- Negative Phase ---------------------------
        # -- computes P(v|h) : Equation at the end of Section 2.3 in the paper
        print("Negative phase ...")
        log+="Negative phase ...\n"
        #timer = utils.Timer('negative phase')
        #with timer:  # measures the time
        # perform the following Gibbs sampling steps, CD_steps times
        for step_idx in range(self.CD_steps):
            # -- compute the negative data given the hidden layer
            self.neg_data[:, :, :, :] = 0
                        
            states =   self.bases.pos_states
            temp = torch.nn.functional.conv2d(states, self.bases.Wb.flip((2,3)), padding=self.btmup_window_shape[0]-1,groups=self.num_bases )
            
            
            self.neg_data= temp.sum(1) + self.vbias
            self.neg_data= self.neg_data[:,np.newaxis,:,:]
            self.bases.neg_sample()

        # -- compute the error as Euclidean distance between positive and
        
        # negative data
        pos_img= self.pos_data[:,:,:,:].numpy()
        neg_img= self.neg_data[:,:,:,:].numpy()
        #err = np.sum( (utils.normalize_image(pos_img, 0, 1) -
        #                            utils.normalize_image(neg_img, 0, 1))**2)
        #self.epoch_err.append(np.sqrt(err))
        err = np.mean( (utils.normalize_image(pos_img, 0, 1) -
                                    utils.normalize_image(neg_img, 0, 1))**2)
        self.epoch_err.append(np.sqrt(err))        
        print("Mean error so far: {0:.3f}".format(np.mean(self.epoch_err)))  
        log+="Mean error so far: {0:.3f}".format(np.mean(self.epoch_err))
        return log

# -----------------------------------------------------------------------------------------------------
    def contrastive_divergence(self, visible_nodes):
        """ Update Layer
        
        Performs Gibbs sampling of the layer's state variables (given the
        previous layer), and then updates the parameters and weights
        accordingly. Here is the steps performed:
            1) Sample each base given 
        
        Input:
            layer_to_learn -- index of the layer to be learned
        
        """
        
        log= self.gibbs(visible_nodes)                

        # -- update the bases only if this layer is being currently        
        # learned
        self.bases.bias_inc[:] = 0
        self.bases.Wb_inc[:,:,:,:] = 0
        self.bases.update()
                
        
        self.vbias_inc = 0
        # -- update the visible layer  bias
        print("Update phase ...")
        # Gradient Descent change
        dV_GD = torch.mean( (self.pos_data - self.neg_data).sum(0) )
        self.vbias_inc = self.epsilon/self.batch * dV_GD
        self.vbias += self.vbias_inc
        
        # print the current state variables of the layer
        log= self.print_statistics(log)
        return log

#-----------------------------------------------------------------------------
    def print_statistics(self,log):
        """ Print Statistics
        Prints the current state variables of the network, including sparsity
        of units' activation, length and change of the weight vector, hidden
        and visible baises, as well as length of the positive and negative data
        vectors.
        """
        
        W_sum = 0
        Winc_sum = 0
        Hb_sum = 0
        Hbinc_sum = 0
        S_sum = 0
        # -- update the bases only if this layer is being currently
        # learned
        
        W_sum += torch.sum(self.bases.Wb ** 2)
        Winc_sum += torch.sum(self.bases.Wb_inc ** 2)
        S_sum += torch.sum(self.bases.pos_states)
        Hb_sum += torch.sum(self.bases.bias ** 2)
        Hbinc_sum += torch.sum(self.bases.bias_inc ** 2)
        
        #print(torch.sqrt(W_sum),W_sum)
        #print(torch.sqrt(Winc_sum),Winc_sum)
        num_units = self.num_bases * self.hidden_shape[0] *self.hidden_shape[1]
        print ("Sparsity measure: {0:.2f} percent".format(100 * float(S_sum)/num_units))
        log+="\nSparsity measure: {0:.2f} percent\n".format(100 * float(S_sum)/num_units)
        print ("||W|| = {0:.2f} ||dW|| = {1:.5f}".format(torch.sqrt(W_sum), torch.sqrt(Winc_sum)))
        log+="||W|| = {0:.2f} ||dW|| = {1:.5f}\n".format(torch.sqrt(W_sum), torch.sqrt(Winc_sum))
        print ("||Hb|| = {0:.2f}  ||dHb|| = {1:.5f}".format(torch.sqrt(Hb_sum), torch.sqrt(Hbinc_sum)))
        log+="||Hb|| = {0:.2f}  ||dHb|| = {1:.5f}\n".format(torch.sqrt(Hb_sum), torch.sqrt(Hbinc_sum))
        print ("||Vb|| = {0:.5f}  ||dVb|| = {1:.6f}".format(abs(self.vbias), abs(self.vbias_inc)))
        log+="||Vb|| = {0:.5f}  ||dVb|| = {1:.6f}\n".format(abs(self.vbias), abs(self.vbias_inc))
        
        return log
#-----------------------------------------------------------------------------
    def weights_for_visualization(self, tile_shape, dir_path="./", save=False):
        """ Visualize Weights
        
        Prepares a visualization array for the bottom-up weights of the bases
        in the layer
        
        Input:
            channel -- index of the channel to whose corresponding weights will
                        be shown
            tile_shape -- shape used to arrange the values for different bases
        Output:
             all_weights -- 2D array containing visualization of weights for 
                            the specified channel of each base in the shape 
                            tile_shape
        
        """
        
        w_size = self.bases.Wb.shape[2]*self.bases.Wb.shape[3]
        all_weights = np.zeros((self.num_bases, w_size))
        
        for i in range(all_weights.shape[0]):
                all_weights[i] = np.reshape(self.bases.Wb[i,0,:, :].numpy(), w_size)
            
        img_shape = (self.bases.Wb.shape[2], self.bases.Wb.shape[3])
        all_weights = utils.tile_raster_images(all_weights, img_shape, tile_shape, tile_spacing = (1,1))
        all_weights = utils.normalize_image(all_weights)

        if save:
            # -- save the visualization array to a PNG file
            filename = dir_path + "/weights.jpg"
            #img = toimage(all_weights)
            #img.save(filename)
            im = Image.fromarray(all_weights).convert('L')
            im.save(filename)

            #if config.DEBUG_MODE:
                #img.show()

            print("Weights were saved to", filename)
            
        return all_weights

#-----------------------------------------------------------------------------
    def output_for_visualization(self, tile_shape, tile_spacing):
        """ Visualize Outputs
        
        Prepares a visualization array for the output of the bases in the 
        layer (taken from the pooling units)
        
        Input:
            tile_shape -- shape used to arrange the values for different bases
            tile_spacing -- number of space to put in between tiles of the
                            output array to make neighboring tiles 
                            distinguishable
        Output:
             ret_array -- 2D array containing visualization of outputs
                          for each base in the shape tile_shape
        
        """
        
        size = list(self.bases.pooling_units[0].size())
        all_outputs = np.zeros((self.num_bases, size[1], size[2]))

        for i in range(all_outputs.shape[0]):
            all_outputs[i] = self.bases.pooling_units[0,i,:,:].numpy()
            
        img_shape = self.bases.pooling_units[0,0].shape
        all_outputs = utils.tile_raster_images(all_outputs, img_shape, tile_shape, tile_spacing)
        all_outputs = utils.normalize_image(all_outputs)
        return all_outputs
    
    
#-----------------------------------------------------------------------------
    def visualize_to_files(self, tile_shape, dir_path):
        """
        Saves the weight vector, and filters to files as images. More images
        can easily be added if needed.
        """
        
        print("Saving to file")
        self.weights_for_visualization(tile_shape, dir_path, save=True)


        # -- visualize the positive data
        # -- visualize the negagive data
      
        img = Image.fromarray(utils.normalize_image(self.pos_data[0,0,:, :].numpy())).convert('L')
        filename = dir_path + "/pos_data.jpg"
        img.save(filename)
            
        img = Image.fromarray(utils.normalize_image(self.neg_data[0,0,:, :].numpy())).convert('L')
        filename = dir_path + "/neg_data.jpg"
        img.save(filename)
        
        filename = dir_path + "/-poooling.jpg"
        all_outputs = self.output_for_visualization(tile_shape, tile_spacing = (1,1))
        img = Image.fromarray(all_outputs).convert('L')
        img.save(filename)

# -----------------------------------------------------------------------------
    def save(self, fname):
        """ Pickle Network
        Saves the entire network in a file, using Python's pickling tools
        Input:
            fname -- name of the file to save the network to
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
            
