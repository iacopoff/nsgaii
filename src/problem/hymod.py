import os,sys

sys.path = ["/home/iff/research/dev/nsgaii/vic"] + sys.path

import numpy as np
import time
from pathlib import Path
from shutil import rmtree
import numpy as np
from edit_config import edit_vic_global,edit_routing_config_dev
import matplotlib.pyplot as plt
from dask.distributed import get_worker
from shutil import copytree,ignore_patterns
import subprocess
from format_soil_params import NSGAII_format_soil_params,format_soil_params
from run_routing import *
from cal_spotpy_functions import read_clusters,set_interaction,pickling,_parseConfig,_readFromFile,create_sampling_gridcells
import glob
from datetime import datetime
import pdb
import pandas as pd

from models.hymod_model import hymod

from eval_functions import kge



def read_config(configFile):

    config_global = _readFromFile(configFile)
    vic_config = _parseConfig(config_global)['hymod_config']

    config = Config(**vic_config)

    return config


class Config:
    def __init__(self,parallel,dataDir,modelDir,inputFile,evalVar):
        self.parallel = parallel
        self.dataDir = dataDir
        self.modelDir = modelDir
        self.inputFile = inputFile  
        self.evalVar = [i for i in evalVar.split(",")]
        self.n_var = 2



class HYMOD:

    def __init__(self,config):
        self.config = config


    def init_evaluation(self):
        data = pd.read_csv(os.path.join(self.config.dataDir,self.config.inputFile),sep=';',parse_dates=["Date"])

        data.dropna(inplace=True)

        data.columns = ["date","prec","TURC","dis"]
        self.PET,self.Precip   = data['TURC'].values,data['prec'].values
        self.date,self.trueObs = data['date'].values,data['dis'].values
        
        
        self.evaluation = data['dis'].values


    def run_simulation(self,x,l):
        self.Factor = 1.783 * 1000 * 1000 / (60 * 60 * 24) 
        #Load Observation data from file

        data = hymod(self.Precip, self.PET, x[0], x[1], x[2], x[3], x[4])
        
        data = np.asarray(data)
        sim = data*self.Factor

        self.simulation = sim

        return sim

    def return_objective_functions(self,simulation,evaluation):

  
        obj = kge(simulation,evaluation).squeeze() #* np.array([-1,-1,1,1])

        def correct(x):
            if x > 0: return x*1
            if x < 0: return x*-1
            return x


        obj = obj[[1,-1]]
        # kge
        obj[0] = obj[0] *-1
        # r
        #obj[1] = obj[1] *-1
        #alpha
        obj[1] = obj[1] * correct(obj[1]) 
        #beta
        #obj[3] = obj[3] * correct(obj[3]) 

        #obj = obj[[1,2,3,0]]

        



        return obj

    def get_evaluation(self):
        return self.evaluation


    def evaluate(self,x,l):
        "x,l are parameters stored in Pop and used in NSGAII"
        sim = self.run_simulation(x,l)

        eva = self.get_evaluation()

        obj = self.return_objective_functions(sim,eva)

        return [obj,sim]

    


    