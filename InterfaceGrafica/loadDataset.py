# -*- coding: utf-8 -*-

import sys

sys.path.append('../Dataset/Semeion/')
import semeion
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy.io as sio
import tensorflow as tf

class loadDataset:
    @staticmethod
    def read(name_dataset):
        if name_dataset == "Semeion":
            file = open('../Dataset/Semeion/semeion.data', 'r')
            lines = file.readlines()

            width = 16
            height = 16
            size = width * height
            classes = 10

            images = []
            labels = []
            decod_labels=[]
            fnumber = 0

            for line in lines:
                data = line.split(' ')
                image = [];
                label = [];

                for i in range(0, size):
                    image.append(int(float(data[i])))
                images.append(np.array(image))
		
                for i in range(size, size + classes):
                    label.append(int(float(data[i]))) 
                labels.append(np.array(label))

                fnumber += 1

            for label in labels:
                result = np.where( label == 1 )
                decod_labels.append( result[0][0] )
    
            X,Y = np.array(images).reshape(len(images),width,height), np.array(decod_labels)
            X_t, X_test, Y_t, Y_test = train_test_split(X,Y,stratify=Y,test_size=0.20, random_state=42)
            X_train, X_valid, Y_train, Y_valid = train_test_split(X_t,Y_t,stratify=Y_t,test_size=0.20, random_state=42)
        
        if name_dataset == "Caltech 101 Silhouettes":
            caltech = sio.loadmat('../Dataset/Caltech/caltech101_silhouettes_28_split1.mat')
            X_train = caltech['train_data'].reshape(4100,28,28)
            X_valid = caltech['val_data'].reshape(2264,28,28)
            X_test = caltech['test_data'].reshape(2307,28,28)
        
        if name_dataset == "MNIST":
            mnist = tf.keras.datasets.mnist
            (X, Y),(X_test, Y_test) = mnist.load_data()
    
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, stratify =Y, test_size=0.20, random_state=42)

            X_train = X_train.astype(np.float)/255
            X_valid = X_valid.astype(np.float)/255
            X_test = X_test.astype(np.float)/255
            
            
        if name_dataset == "MPEG":
            df = pd.read_csv('../Dataset/MPEG/MPEG.csv')
            X = df.values[:,1:].reshape(1402,28,28)
            #normalizing
            X = np.around( X.astype(np.float)/255 )
            
            X_t, X_test = train_test_split(X, test_size=0.20, random_state=42)
            X_train, X_valid = train_test_split(X_t,test_size=0.20, random_state=42)
            
        
        return X_train, X_valid, X_test 
    