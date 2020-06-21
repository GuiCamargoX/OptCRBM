# -*- coding: utf-8 -*-
import operator
from random import randint,uniform,seed,random
import numpy as np
import pandas as pd
import utils
import os

import sys
sys.path.append('../CRBM/')
from crbm import CRBM
from loadDataset import loadDataset
from PandasLoadPyqt5 import PandasModel
from dialog2 import Ui_Dialog as Form

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from PyQt5 import QtWidgets

class  EvolutionaryAlgorithm:
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
    
    def __init__(self, hyperameters, bound, parameters_ea, dataset, method):
        self.model = hyperameters
        self.bound = bound
        self.parameters_ea= parameters_ea
        self.dataset = dataset
        self.method= method
        #Dataset
        self.X_train, self.X_valid, self.X_test = loadDataset.read(self.dataset)
        
    def run(self, progress, progress_bar, tableResults):
        self.progressArea2 = progress
        self.progressBar2 = progress_bar
        self.progressBar2.setMinimum(0)
        self.progressBar2.setMaximum(self.model['epoch_per_layer'])
        
        if self.method=="Genetic Programming":
            self.run_GP()
        
        if self.method=="Genetic Algorithm":
            self.run_GA()
            print('teste')

        seed(318)
    
        pop = self.toolbox.population(n=self.parameters_ea["n_pop"])
        hof = tools.HallOfFame(1 , lambda ind,hofer: ind.fitness.values[0]== hofer.fitness.values[0] )
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
    
        pop, log_ea = algorithms.eaSimple(pop, self.toolbox, cxpb=self.parameters_ea["p_cx"], mutpb=self.parameters_ea["p_mut"], ngen=self.parameters_ea["n_gen"], stats=stats,
                                       halloffame=hof, verbose=True)
        
        
        best_param = hof[0]
        if self.method=="Genetic Programming":
            best_param = self.applyBound( self.toolbox.compile(hof[0]) )            
        
        self.model['num_bases']= best_param[0]
        self.model['btmup_window_shape']= ( best_param[1], best_param[1]) 
        self.model['CD_steps']=best_param[2]
    
        # -------------------------- Read the Input Images ------------------------
        torch_batch = utils.process_array_to_pytorch(self.X_train, self.model['btmup_window_shape'], self.model['block_shape'] )
        torch_batch_test = utils.process_array_to_pytorch(self.X_test, self.model['btmup_window_shape'], self.model['block_shape'] )
    
        print("Simulation starts with an unlearned network with random weights..\n")
        network = CRBM(self.model, (torch_batch.shape[2],torch_batch.shape[3], torch_batch.shape[1] ) )
        
        self.progressBar2.setValue(0)
        for epoch_idx in range(self.model['epoch_per_layer']):
            print("Training trial #%s.." % epoch_idx)
    
            self.progressArea2.append("\n\n\nTraining error")
            for i in range(len(torch_batch)//network.batch):
                print("\n------ Epoch ", epoch_idx, ", batch " , i,"------")
                b= torch_batch[i*self.model['batch']:i*self.model['batch']+self.model['batch'],:,:,:]
                log= network.contrastive_divergence(b)
                self.progressArea2.append("\n------ Epoch "+ str(epoch_idx) + ", batch "+ str(i)+ "------")
                self.progressArea2.append(log)
                QtWidgets.QApplication.processEvents()
            
            # -- compute mean of error made in the epoch and save it to the file
            mean_err_train = np.mean(network.epoch_err)
            # flush the errors made in the training
            network.epoch_err = []
            
            print("\n\n\nERROR Test")
            self.progressArea2.append("\n\n\nTest error")
            for i in range(len(torch_batch_test)//network.batch):
                print("\n------ Epoch ", epoch_idx, ", batch " , i,"------")
                b= torch_batch_test[i*self.model['batch']:i*self.model['batch']+self.model['batch'],:,:,:]
                log= network.gibbs(b)
                self.progressArea2.append("\n------ Epoch "+ str(epoch_idx) + ", batch "+ str(i)+ "------")
                self.progressArea2.append(log)
                QtWidgets.QApplication.processEvents()
            
            # -- compute mean of error made in the epoch and save it to the file
            mean_err_test = np.mean(network.epoch_err)
            # flush the errors made in the previous epoch
            network.epoch_err = []
                    
            
            # -- stop decaying after some point
            # TEST
            if network.std_gaussian > network.model['sigma_stop']:
                 network.std_gaussian *= 0.99
                 
            self.progressBar2.setValue(epoch_idx)
        
        self.progressBar2.setValue(self.model['epoch_per_layer']) 
        # -- create a directory to contains the results (oputputs) of the
        # simualtion, if it doesn't exist already
        if not os.path.exists(self.misc_params['results_dir']):
            os.mkdir(self.misc_params['results_dir'])             
        # -- visualize layers and save the network at the end of each epoch
        network.visualize_to_files( (5, 5), dir_path=self.misc_params['results_dir'])
        network.save(self.misc_params['results_dir'] + self.misc_params['pickle_fname'])
        
        data = [[i for i in item.values()] for item in log_ea]
    
        df = pd.DataFrame(data, columns=log_ea.header)
        df.drop(columns=['nevals'], inplace=True)
        
        model = PandasModel(df)
        tableResults.setModel(model)
        
        
        Dialog = QtWidgets.QDialog()
        ui = Form()
        ui.setupUi(Dialog)
        ui.set_groups_result( str(best_param[0]) )
        ui.set_filtersize_result('{} X {}'.format(best_param[1],best_param[1]) )
        ui.set_cdsteps_result(str(best_param[2]))
        ui.set_training_error_result( "{0:.10f}".format( mean_err_train))
        ui.set_valid_error_result("{0:.10f}".format(hof[0].fitness.values[0] ))
        ui.set_test_error_result("{0:.10f}".format(mean_err_test))
        ui.set_curve_learning(df)
        Dialog.exec_()
        Dialog.show()
        
    
    def run_GA(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Structure initializers
        self.toolbox = base.Toolbox()
        self.toolbox.register("hyperparameters", lambda:[ randint(self.bound["num_bases"][0],self.bound["num_bases"][1]),
                                                      randint(self.bound["btmup_window_shape"][0],self.bound["btmup_window_shape"][1]),
                                                      randint(self.bound["CD_steps"][0],self.bound["CD_steps"][1])
                                                      ] )
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.hyperparameters )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutBound) 
        self.toolbox.register("evaluate", self.evalAcc )
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        
    def mutBound(self,individual, indpb=0.5):
        i=0
        
        for key,val in self.bound.items():
            if random() < indpb:
                if val[2] == int:
                   individual[i] = randint(val[0],val[1])
                if val[2] == float:
                   individual[i] = uniform(val[0],val[1])
            i+=1    
    
        return individual,


    def run_GP(self):
        
        pset = gp.PrimitiveSet("main", 0)
        pset.addPrimitive(np.add, arity=2)
        pset.addPrimitive(np.subtract, arity=2)
        pset.addPrimitive(np.negative, arity=1)
        pset.addEphemeralConstant("rand101", lambda: [randint(self.bound["num_bases"][0],self.bound["num_bases"][1]),
                                              randint(self.bound["btmup_window_shape"][0],self.bound["btmup_window_shape"][1]),
                                              randint(self.bound["CD_steps"][0],self.bound["CD_steps"][1])
                                              ] )

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Attribute generator
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

        # Structure initializers
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)
        
        
        self.toolbox.register("evaluate", self.evalAcc )
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
        
        #limit the height of generated individuals. Avoid bloat. Koza suggest max depth of 17
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


    def evalAcc(self,individual ):
        
        if self.method=="Genetic Programming":
        # Compute the tree expression
            result = self.toolbox.compile(expr=individual)
            param = self.applyBound(result)
        if self.method=="Genetic Algorithm":
            #print(individual)
            param = [abs(p) for p in individual]


        self.model['num_bases']= param[0]
        self.model['btmup_window_shape']= ( param[1], param[1]) 
        self.model['CD_steps']=param[2]

        # -------------------------- Read the Input Images ------------------------
        torch_batch = utils.process_array_to_pytorch(self.X_train, self.model['btmup_window_shape'], self.model['block_shape'] )
        torch_batch_valid = utils.process_array_to_pytorch(self.X_valid, self.model['btmup_window_shape'], self.model['block_shape'] )

        print("Simulation starts with an unlearned network with random weights..\n")
        network = CRBM(self.model, (torch_batch.shape[2],torch_batch.shape[3], torch_batch.shape[1] ) )
    
        self.progressBar2.setValue(0)
        for epoch_idx in range(self.model['epoch_per_layer']):
            print("Training trial #%s.." % epoch_idx)

            self.progressArea2.append("\n\n\nTraining error")
            for i in range(len(torch_batch)//network.batch):
                print("\n------ Epoch ", epoch_idx, ", batch " , i,"------")
                b= torch_batch[i*self.model['batch']:i*self.model['batch']+self.model['batch'],:,:,:]
                log= network.contrastive_divergence(b)
                self.progressArea2.append("\n------ Epoch "+ str(epoch_idx) + ", batch "+ str(i)+ "------")
                self.progressArea2.append(log)
                QtWidgets.QApplication.processEvents()
                

            # flush the errors made in the training
            network.epoch_err = []
        
            print("\n\n\nERROR Test")
            self.progressArea2.append("\n\n\nTest error")
            for i in range(len(torch_batch_valid)//network.batch):
                print("\n------ Epoch ", epoch_idx, ", batch " , i,"------")
                b= torch_batch_valid[i*self.model['batch']:i*self.model['batch']+self.model['batch'],:,:,:]
                log= network.gibbs(b)
                self.progressArea2.append("\n------ Epoch "+ str(epoch_idx) + ", batch "+ str(i)+ "------")
                self.progressArea2.append(log)
                QtWidgets.QApplication.processEvents()
        
            # -- compute mean of error made in the epoch and save it to the file
            mean_err = np.mean(network.epoch_err)
            # flush the errors made in the previous epoch
            network.epoch_err = []
                
        
            # -- stop decaying after some point
            # TEST
            if network.std_gaussian > network.model['sigma_stop']:
                network.std_gaussian *= 0.99    
            
            self.progressBar2.setValue(epoch_idx)
        
        self.progressBar2.setValue(self.model['epoch_per_layer']) 
        return mean_err,



    def applyBound(self,value):#interna retorna lista
        params = np.absolute(value).tolist()
        i=0
        
        for key,val in self.bound.items():
            if isinstance(val[0], int) and isinstance(val[1], int):
                params[i] = int(params[i])
            #else :
                #print(params[i])
            params[i] = val[0] if params[i] < val[0] else params[i]
            params[i] = val[1] if params[i] > val[1] else params[i]
            #params[i]= val[0]+ (params[i] % (val[1]-val[0]-1) )
            i+=1
    
        return params 
