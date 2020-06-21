#import collections
import numpy as np
#import dataset
#from random import shuffle
#from sklearn.model_selection import train_test_split


def read_data_semeion(fname = '../../Dataset/Semeion/semeion.data'):
    file = open(fname, 'r')
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
		#if fnumber < 10:
			#image_utils.show(image, width, height)
		#image_utils.save('./dataset/semeion/images/' + str(fnumber) + '.png', array, width, height)

    for label in labels:
        result = np.where( label == 1 )
        decod_labels.append( result[0][0] )
    
    
    #X_train, X_test, y_train, y_test = train_test_split( images, decod_labels, test_size=0.20, random_state=42, stratify= decod_labels )
    
    return np.array(images).reshape(len(images),width,height), np.array(decod_labels)

#read_data_semeion()
