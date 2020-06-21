#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" image_data module

All the operations performed on image data

"""

import os
from PIL import Image
import scipy as sp
import numpy as np 

from numpy.fft import fft2, ifft2, fftshift
import math
import torch

#import pylab as pl   # matplotlib

import utils

class ImageData:
    
    def __init__(self, image_params):
        self.extentions = image_params['EXTENSIONS']
        self.num_channels = image_params['num_channels']
        self.max_image_edge = image_params['max_image_edge']
        self.img_path = image_params['image_path']
        self.samples_per_image = image_params['samples_per_image']
        self.image_patch_shape = image_params['image_patch_shape']
        
        self.w_shape = image_params['window_shape']
        self.block_shape = image_params['block_shape']
        
        
        trim_shape= utils.trim_array_maxpool(arr_shape=self.image_patch_shape,
                                                     conv_window_shape=self.w_shape,
                                                     pooling_shape=self.block_shape)
        
        self.img_shape= (trim_shape[0], trim_shape[1], self.num_channels)
        self.images = []  # list of images to be loaded
        self.curr_input = None        
    
    def load_images(self):
        """ Load all the image files recursively

        Input:
          image_params -- the params to load the images
        
        Output:
          appends all the images to globals.images
        
        """        
        # -- convert to absolute path and verify the path
        img_path = os.path.abspath(self.img_path)
        print("Image source:", img_path)
        if not os.path.isdir(img_path):
            raise ValueError("%s is not a directory" % (img_path))
        
        # -- extract the file names
        tree = os.walk(img_path)
        filelist = []
        #categories = tree.next()[1]    
        for path, dirs, files in tree:
            if dirs != []:
                msgs = ["invalid image tree structure:"]
                for d in dirs:
                    msgs += ["  "+"/".join([root, d])]
                msg = "\n".join(msgs)
                raise Exception(msg)
            filelist += [ path+'/'+f for f in files if os.path.splitext(f)[-1] in self.extentions ]
        filelist.sort()    
        
        #utils.visualize_array(filelist[0])
        # -- load and preprocess images
        for img_fname in filelist: #[0:1]:
            img = self.load_process_image(img_fname)
            self.images.append(img)
            #utils.visualize_array(img)
        
        #self.create_batch(weight_shape,block_shape)
        #print len(categories), "categories found:"
        #print categories
    
    
    # -----------------------------------------------------------------------------
    def create_batch(self):
        
        num_images = len(self.images)
        batch = torch.FloatTensor(num_images*self.samples_per_image, self.num_channels, self.img_shape[0], self.img_shape[1])
        curr_input = np.zeros(self.img_shape) 
        count=0

        for img_idx in range(num_images):
            for patch_idx in range(self.samples_per_image):
                
                # -- get an image patch and trim it so the size fits for convolution
                img_patch = self.get_image_patch(img_idx, self.image_patch_shape)
                
                for cnl in range(self.num_channels):
                    curr_input[:, :, cnl] = utils.trim_array_maxpool(arr=img_patch[:, :, cnl], conv_window_shape=self.w_shape, pooling_shape=self.block_shape)
                  
    
                #Defining channel (N, C, H, W) format. N is the number of samples/batch_size. 
                #C is the channels. H and W are height and width resp.
                img_format_tensor=np.rollaxis(curr_input, 2, 0)
                #add one more dimensional as amount of batch
                img_format_tensor= torch.from_numpy(img_format_tensor[np.newaxis,:]).float() 
                batch[count]= img_format_tensor
                count+=1
                
        return batch
                
    # -----------------------------------------------------------------------------
    def load_process_image(self, img_fname):
        """ Return a preprocessed image as a numpy array

        Inputs:
          img_fname -- image filename

        Outputs:
          imga -- result

        """

        print("loading "+img_fname.split('/')[-1])
        # -- open image
        img = Image.open(img_fname)
        
        """if self.resize is not None:
            self.max_image_edge = 
            img = img.resize((self.resize[0],self.resize[1]))
        """
        
        print("preprocessing "+img_fname.split('/')[-1]+" ...")
        # -- resize and whiten image
        img = self.preprocess(img)
        
        return img
        
    # -----------------------------------------------------------------------------
    def preprocess(self, img):
        """ Return a resized and whitened image as a numpy array
        
        The following steps are performed:
        
        1) The image is resized so the longest edge is of size max_edge, while 
        keeping the width-height ratio intact.
        2) The resized image is whitened using the Olshausen & Field (1997) 
        alogorithm.
        
        Inputs:
          img -- image in python Image format
          max_edge -- maximum edge length

        Outputs:
          imga -- result

        """
        
        iw, ih = img.size
        
        if isinstance(self.max_image_edge, tuple):
            img = img.resize((self.max_image_edge[0],self.max_image_edge[1]))
        else:
            # -- resize so that the biggest edge is max_edge (keep aspect ratio)
            if iw > ih:
                new_iw = self.max_image_edge
                new_ih = int(math.floor(1.* self.max_image_edge * ih/iw))
            else:
                new_ih = self.max_image_edge
                new_iw = int(math.floor(1.* self.max_image_edge * iw/ih))
            
            img = img.resize((new_iw, new_ih), Image.BICUBIC)


        # -- convert the image to greyscale (mode "L" for luminance)
        if img.mode != 'L':
            img = img.convert('L')
        
        # -- perform olshausen whitening on the image
        imga = self.olshausen_whitening(img)

        
        # -- perform some extra normalization
        imga -= np.mean(imga)
        imga /= math.sqrt(np.mean(imga**2))
        imga *= math.sqrt(0.1) # for some tricks?! :TODO
        #imga = utils.normalize_image(imga, -0.01, 0.01)
        return imga

    # -----------------------------------------------------------------------------
    def olshausen_whitening(self, img):
        """ Return a whitened image as a numpy array
        
        Performs image whitening as described in Olshausen & Field's Vision 
        Research article (1997)

        f_0 controls the radius of the decay function. 
        n controls the steepness of the radial decay.

        Input:
          img -- image in python Image format

        Outputs:
          img -- result image in numpy array format

        """
        iw, ih = img.size
        # -- The different color channels are stored in the third dimension, 
        # such that a grey-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
        img = np.array(img, dtype=np.float)  # convert image to numpy array
        
        print(img.shape,'\n')
        # Let all images be 3D (MxNxC, where C is number of channels) for consistency
        img = img.reshape(img.shape[0], img.shape[1], self.num_channels)
        
        img = (img - np.mean(img)) / np.std(img)

        X, Y = np.meshgrid(np.arange(-iw/2, iw/2), np.arange(-ih/2, ih/2))
        
        f_0 = 0.4 * np.mean([iw,ih])  # another source used min
        n = 4
        rho = np.sqrt(X**2 + Y**2)
        
        filt = rho * np.exp(-(rho/f_0)**n)  # low_pass filter
        
        for cnl in range(self.num_channels):
            img[:, :, cnl] = ifft2(fft2(img[:, :, cnl]) * fftshift(filt)).real

        img /= np.std(img)
        
        #print img[0:9, 0:9]
        #tt = Image.frombuffer('L', (iw, ih), img)
        #tt.show()
        
        
        return img

    # -----------------------------------------------------------------------------
    def get_image_patch(self, img_index, patch_shape):
        """ Return a random patch of the image (preprocessed) with the given index
        
        Inputs:
          img_index -- index of the image to be used
          patch_shape -- shape of the image patch to be returned

        Outputs:
          img_patch -- path from the image with the given size

        """
        rows, cols = self.images[img_index].shape[0], self.images[img_index].shape[1]
        
        cx = np.random.randint(rows - patch_shape[0]) 
        cy = np.random.randint(cols - patch_shape[1])
        
        image_patch = self.images[img_index][cx:cx+patch_shape[0], cy:cy+patch_shape[1], :]
        

        # -- preprocess image_patch
        image_patch -= np.mean(image_patch)

        if np.random.rand() > 0.5:
            image_patch = np.fliplr(image_patch)


        return image_patch


   
