import os,sys

sys.path = ["/home/iff/research/dev/nsgaii/vic"] + sys.path

import numpy as np
import time
from pathlib import Path
from shutil import rmtree
import numpy as np
from edit_config import edit_vic_global,edit_routing_config_dev

from dask.distributed import get_worker
from shutil import copytree,ignore_patterns
import subprocess
from format_soil_params import NSGAII_format_soil_params,format_soil_params
from run_routing import *
from cal_spotpy_functions import read_clusters,set_interaction,pickling,_parseConfig,_readFromFile,create_sampling_gridcells
import glob
from datetime import datetime
import xarray as xr
import pandas as pd
import pickle

from eval_functions import kge

def read_config(configFile):

    config_global = _readFromFile(configFile)
    vic_config = _parseConfig(config_global)['vic_config']

    config = Config(**vic_config)

    return config


class Config:
    def __init__(self,parallel,parentDir,scriptDir,paramFile,vicOutVar,evalDir,evalFile,evalVar,
                  calStart,calEnd,timeStep,calOutName):
        self.parallel = parallel
        self.parentDir = parentDir
        self.scriptDir = scriptDir
        self.paramFile = paramFile
        self.vicOutVar = [i for i in vicOutVar.split(",")]
        self.evalDir = evalDir
        self.evalFile = [i for i in evalFile.split(",")]

        self.evalVar = [i for i in evalVar.split(",")]
        self.calStart = datetime.strptime(calStart,'%Y-%m-%d')
        self.calEnd = datetime.strptime(calEnd,'%Y-%m-%d')
        self.timeStep = timeStep
        self.calOutName = calOutName
        self.n_var = len(self.vicOutVar)



class VIC:

    def __init__(self,config):
        self.config = config

        self.gridcells = create_sampling_gridcells(config.paramFile)

        self.counter = 0
        self.n_gridcells =len(self.gridcells)
        self.flagnan = False

        self.evalDatasetN = len(self.config.evalFile)

        self.simulation_storage = []

    def init_evaluation(self):
        # open files
        print("opening evaluation")


        eval = {}
        for i,file in enumerate(self.config.evalFile):
            eval[file] = xr.open_dataset(self.config.evalDir + "/" + file,autoclose=True)[self.config.evalVar[i]]
            eval[file]["time"] = eval[file].indexes['time'].normalize()
            eval[file] = eval[file].loc[self.config.calStart:self.config.calEnd] # eval[file].loc[self.datastart:(self.dataend + datetime.timedelta(days=1))]


        date = pd.date_range(self.config.calStart, self.config.calEnd)

        outer = {}
        for ieval in eval:
            out = {}
            for gr in self.gridcells.index:
                s = self.gridcells.loc[gr]
                out[gr] = eval[ieval].sel(lat=s.lat, lon=s.lon).values
            evaldf = pd.DataFrame(out)
            evaldf["time"] = date
            evaldf  = evaldf.set_index("time")
            outer[ieval] = evaldf#.rolling(10).mean().dropna()

        if self.config.timeStep == "M":
            # monthly calibration



            print("COUNTER:")
            print(self.counter)

            obsSeries = []
            for ix, iout in enumerate(outer):
                obsSeries.append([outer[iout].resample("M").agg(np.nanmean).iloc[:, i].values for i in range(len(self.gridcells))])

            #import pdb; pdb.set_trace()

            #self.time_series_length = len(obsSeries[0][0])

            self.index = []
            self.keepindex = []
            if self.counter <1:
                print("remove index when counter <1")
                for ix,ts in enumerate(obsSeries[0]):
                    if np.any(np.isnan(ts)):
                        self.flagnan = True
                        self.index.append(ix)
                    else:
                        self.keepindex.append(ix)



                for i in range(self.evalDatasetN):
                    obsSeries[i] = [arr for i,arr in enumerate(obsSeries[i]) if i not in self.index]

            if self.flagnan:
                self.n_gridcells = len(self.gridcells.iloc[self.keepindex, :])

                pickle_out = open(f"{self.config.parentDir}/gridcell_index.pickle","wb")

                pickle.dump(self.keepindex, pickle_out)
                pickle_out.close()

            elif self.counter >=1 and self.flagnan:
                obsSeries[0] = obsSeries[0][self.index]

                print("removing index nan when counter >1------------------------")
                print(self.index)

            self.savedgridID = self.gridcells.iloc[self.keepindex].index.values

            self.counter = self.counter + 1


        if self.config.timeStep == "D":
            # daily calibration
            obsSeries = []
            for ix, iout in enumerate(outer):
                obsSeries.append([outer[iout].iloc[:, i].values for i in range(len(self.gridcells))])

            self.savedgridID = self.gridcells.index.values


        print("observation length")
        print(len(obsSeries))

        self.evaluation =  obsSeries


    def get_evaluation(self):
        return self.evaluation


    def run_simulation(self,x,l):
        if self.config.parallel == "seq":
            print("need to implement that")
        elif self.config.parallel == "dask":

            print(f"WORKER ID IS: {get_worker().id}")
            workerID = "_".join(get_worker().id.split("-")[:2])

            #workerID = "fake_worker"
            workerDir = os.path.join(self.config.parentDir,workerID)


            copytree(os.path.join(self.config.parentDir,"run"),workerDir,ignore = ignore_patterns('tests','*py','fluxes*'))

            os.chdir(workerDir)

            globalFile_old = glob.glob("global*.txt")[0]
            globalFile_new = globalFile_old.split(".")[0] + "_{}.txt".format(workerID)

            # print("rename param")
            paramFile_old = glob.glob("param*.nc")[0]
            paramFile_new = paramFile_old.split(".")[0] + "_{}.nc".format(workerID)
            os.rename(paramFile_old, paramFile_new)


            edit_vic_global(file=globalFile_old, par_file=paramFile_new, parall_dir=workerDir,
                            globalFile_new=globalFile_new)

            os.remove(globalFile_old)


            # FORMAT NETCDF
            print("format netcdf")
            NSGAII_format_soil_params(nc_file=paramFile_new,
                                  #gridcells=self.gridcells,
                                  param_names =l,
                                  param_values =x)

            print("executing VIC model...")

            try:

                subprocess.check_output(f'vic_image.exe -g {globalFile_new}', shell=True)

            except:
                print("vic run failed")



            # EXTRACT FLUXES

            try:
                file = xr.open_dataset(glob.glob(workerDir  + "/fluxes*.nc")[0], autoclose=True)
                sim = {}
                for ivar in self.config.vicOutVar:
                    if ivar == "OUT_SOIL_MOIST":
                        sim[ivar] = file[ivar][:, 0, ...]  # get surface layer
                    elif ivar == "OUT_EVAP":
                        sim[ivar] = file[ivar]


            except:
                print("can t open sim file")

            date = pd.date_range(self.config.calStart, self.config.calEnd)

            outer = {}
            for isim in sim:
                out = {}
                for gr in self.gridcells.index:
                    s = self.gridcells.loc[gr]
                    out[gr] = sim[isim].sel(lat=s.lat, lon=s.lon).values
                simdf = pd.DataFrame(out)
                simdf["time"] = date
                simdf = simdf.set_index("time")
                outer[isim] = simdf#.rolling(10).mean().dropna()  ## testing rolling mean


            try:
                if self.config.timeStep == "M":
                    # monthly calibration
                    simSeries = []
                    for ix,iout in enumerate(outer):
                        simSeries.append([outer[iout].resample("M").agg(np.nanmean).iloc[:, i].values for i in range(len(self.gridcells))])

                    if self.flagnan:
                        for i in range(self.evalDatasetN):
                            simSeries[i] = [arr for i, arr in enumerate(simSeries[i]) if i not in self.index]

                    print(f"length monthly simulation time series {len(simSeries )}")

            except:
                print("bordello con la version monthly")

            if self.config.timeStep == "D":
                # daily calibration
                simSeries = []
                for ix,iout in enumerate(outer):
                    simSeries.append([outer[iout].iloc[:, i].values for i in range(len(self.gridcells))])

                print(f"length daily simulation time series {len(simSeries )}")



            os.chdir(self.config.parentDir)

            rmtree(workerDir)

        else:
            pass

        self.simulation = simSeries
        return simSeries




    def objective_functions(self,simulation,evaluation):

        obj = np.array([])

        for sim,eva in zip(simulation,evaluation):
            sim = np.mean(np.vstack(sim),axis=0)
            eva = np.mean(np.vstack(eva),axis=0)

            obj = np.append(obj, - kge(sim,eva)[0])

            

        return obj


    def evaluate(self,x,l):

        sim = self.run_simulation(x,l)

        eva = self.get_evaluation()

        obj = self.objective_functions(sim,eva)

        return [obj,sim]

