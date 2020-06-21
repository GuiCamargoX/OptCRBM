# -*- coding: utf-8 -*-
import sys
sys.path.append('../../CRBM/')
from crbm import CRBM
import utils
import scipy.io as sio
import os

from sklearn.model_selection import train_test_split

import random
import numpy as np
from numpy import inf
from random import randint,uniform
import pandas as pd


from deap import algorithms
from deap import base
from deap import creator
from deap import tools


#Param init
model={
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

#Bound (down,up)
bound = {   
       # -- number of bases or "groups" in the layer - equivalent to 
       #       parameter K in the Lee et. al. ICML09 paper 5	    
	    "num_bases" : (1,25,int),
	#shape of the bottom-up filter
            "btmup_window_shape" : (5, 15,int),
            "CD_steps" : (1,3,int)}

### VARIABLES ###
p_cx = 0.4  # Probability of crossover
p_mut = 0.1 # Probability of mutation
n_pop = 30   # Population per generation
n_gen = 6   # Number of generations

#Dataset
caltech = sio.loadmat('../../Dataset/Caltech/caltech101_silhouettes_28_split1.mat')
X_train = caltech['train_data'].reshape(4100,28,28)
X_valid = caltech['val_data'].reshape(2264,28,28)
X_test = caltech['test_data'].reshape(2307,28,28)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Structure initializers
toolbox = base.Toolbox()
toolbox.register("hyperparameters", lambda:[  randint(bound["num_bases"][0],bound["num_bases"][1]),
                                              randint(bound["btmup_window_shape"][0],bound["btmup_window_shape"][1]),
                                              randint(bound["CD_steps"][0],bound["CD_steps"][1])
                                              ] )

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.hyperparameters)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)

def mutBound(individual, vary, indpb):
    i=0
    
    for key,val in vary.items():
        if random.random() < indpb:
            if val[2] == int:
               individual[i] = randint(val[0],val[1])
            if val[2] == float:
               individual[i] = uniform(val[0],val[1])
        i+=1    

    return individual,
toolbox.register("mutate", mutBound, vary=bound , indpb=0.4)

def evalAcc(individual):
    # Compute the tree expression
    param = [abs(p) for p in individual]

    print( param )

    model['num_bases']= param[0]
    model['btmup_window_shape']= ( param[1], param[1]) 
    model['CD_steps']=param[2]

    # -------------------------- Read the Input Images ------------------------
    torch_batch = utils.process_array_to_pytorch(X_train, model['btmup_window_shape'], model['block_shape'] )
    torch_batch_valid = utils.process_array_to_pytorch(X_valid, model['btmup_window_shape'], model['block_shape'] )

    print("Simulation starts with an unlearned network with random weights..\n")
    network = CRBM(model, (torch_batch.shape[2],torch_batch.shape[3], torch_batch.shape[1] ) )
    
    for epoch_idx in range(model['epoch_per_layer']):
        print("Training trial #%s.." % epoch_idx)


        for i in range(len(torch_batch)//network.batch):
            print("\n------ Epoch", epoch_idx, ", batch" , i,"------")
            b= torch_batch[i*model['batch']:i*model['batch']+model['batch'],:,:,:]
            print(i,i*model['batch'],i*model['batch']+model['batch'])
            network.contrastive_divergence(b)
                

        # flush the errors made in the training
        network.epoch_err = []
        
        print("\n\n\nERROR Test")
        for i in range(len(torch_batch_valid)//network.batch):
            print("\n------ Epoch", epoch_idx, ", batch" , i,"------")
            b= torch_batch_valid[i*model['batch']:i*model['batch']+model['batch'],:,:,:]
            print(i,i*model['batch'],i*model['batch']+model['batch'])
            network.gibbs(b)
        
        # -- compute mean of error made in the epoch and save it to the file
        mean_err = np.mean(network.epoch_err)
        # flush the errors made in the previous epoch
        network.epoch_err = []
                
        
        # -- stop decaying after some point
        # TEST
        if network.std_gaussian > network.model['sigma_stop']:
             network.std_gaussian *= 0.99    
        
    return mean_err,

toolbox.register("evaluate", evalAcc )
toolbox.register("select", tools.selTournament, tournsize=3)


import smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders

def send_mail(send_from,send_to,subject,text,server,port,username='',password='',isTls=True):
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to
    msg['Date'] = formatdate(localtime = True)
    msg['Subject'] = subject
    msg.attach(MIMEText(text))

    part = MIMEBase('application', "octet-stream")
    part.set_payload(open("GAinCRBM.xlsx", "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="GAinCRBM.xlsx"')
    msg.attach(part)

    #context = ssl.SSLContext(ssl.PROTOCOL_SSLv3)
    #SSL connection only working on Python 3+
    smtp = smtplib.SMTP(server, port)
    if isTls:
        smtp.starttls()
    smtp.login(username,password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()

def run_GA():

    random.seed(318)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=p_cx, mutpb=p_mut, ngen=n_gen, stats=stats,
                                   halloffame=hof, verbose=True)

    return pop, log, hof[0]

if __name__ == "__main__":
    pop, log, hof = run_GA()    
    #gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
        
    best_param= hof

    model['num_bases']= best_param[0]
    model['btmup_window_shape']= ( best_param[1], best_param[1]) 
    model['CD_steps']=best_param[2]

    # -------------------------- Read the Input Images ------------------------
    torch_batch = utils.process_array_to_pytorch(X_train, model['btmup_window_shape'], model['block_shape'] )
    torch_batch_test = utils.process_array_to_pytorch(X_test, model['btmup_window_shape'], model['block_shape'] )

    print("Simulation starts with an unlearned network with random weights..\n")
    network = CRBM(model, (torch_batch.shape[2],torch_batch.shape[3], torch_batch.shape[1] ) )
    
    for epoch_idx in range(model['epoch_per_layer']):
        print("Training trial #%s.." % epoch_idx)


        for i in range(len(torch_batch)//network.batch):
            print("\n------ Epoch", epoch_idx, ", batch" , i,"------")
            b= torch_batch[i*model['batch']:i*model['batch']+model['batch'],:,:,:]
            print(i,i*model['batch'],i*model['batch']+model['batch'])
            network.contrastive_divergence(b)
        
        # -- compute mean of error made in the epoch and save it to the file
        mean_err_train = np.mean(network.epoch_err)
        # flush the errors made in the training
        network.epoch_err = []
        
        print("\n\n\nERROR Test")
        for i in range(len(torch_batch_test)//network.batch):
            print("\n------ Epoch", epoch_idx, ", batch" , i,"------")
            b= torch_batch_test[i*model['batch']:i*model['batch']+model['batch'],:,:,:]
            print(i,i*model['batch'],i*model['batch']+model['batch'])
            network.gibbs(b)
        
        # -- compute mean of error made in the epoch and save it to the file
        mean_err_test = np.mean(network.epoch_err)
        # flush the errors made in the previous epoch
        network.epoch_err = []
                
        
        # -- stop decaying after some point
        # TEST
        if network.std_gaussian > network.model['sigma_stop']:
             network.std_gaussian *= 0.99
    
    
    # -- create a directory to contains the results (oputputs) of the
    # simualtion, if it doesn't exist already
    if not os.path.exists(misc_params['results_dir']):
        os.mkdir(misc_params['results_dir'])             
    # -- visualize layers and save the network at the end of each epoch
    network.visualize_to_files( (5, 5), dir_path=misc_params['results_dir'])
    network.save(misc_params['results_dir'] + misc_params['pickle_fname'])
        
    data = [[i for i in item.values()] for item in log]
    data.append( hof )
        
    data.append(['error_train', mean_err_train])
    data.append( ['error_valid',hof.fitness ] )
    data.append(['error_test', mean_err_test])
    
    
    df = pd.DataFrame(data, columns=log.header)
    df.to_excel (r'GAinCRBM.xlsx', index = None, header=True) #Don't forget to add '.xlsx' at the end of the path

    send_mail('camargo','guicamargo551@gmail.com','CRBM_Semeion_ga','','smtp.gmail.com', 587,
          'camargoxdg@gmail.com','Goku55736238')
