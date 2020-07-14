# --- module import
import os,sys
sys.path = ["/projects/mas1261/wa_software/spotpy1.5.6"] + sys.path

import xarray as xr
import spotpy
import datetime
import subprocess
import pickle
from shutil import copytree, ignore_patterns
from distutils.dir_util import remove_tree
from edit_config import edit_vic_global,edit_routing_config_dev

import numpy as np
import pandas as pd

from format_soil_params import format_soil_params_distributed,format_soil_params
from run_routing import *
from cal_spotpy_functions import read_clusters,set_interaction,pickling,_parseConfig,_readFromFile

import glob

# --- define spotpy class object

# spotpy setup class: it handles all the calibration
class spotpy_setup(object):
    """
     spot_setup: class
        simulation: function
            Should be callable with a parameter combination of the parameter-function
            and return an list of simulation results (as long as evaluation list)
        parameter: function
            When called, it should return a random parameter combination. Which can
            be e.g. uniform or Gaussian
        objectivefunction: function
            Should return the objectivefunction for a given list of a model simulation and
            observation.
        evaluation: function
            Should return the true values as return by the model.
    """
    def __init__(self,
                 dir_work,
                 dir_scripts,
                 cal_start,
                 cal_end,
                 time_step,
                 params,
                 cal_mode,
                 obj_funct_obj,
                 obj_f_opt_direction,
                 obj_funct_param_flag,
                 obj_funct_param,
                 dfgridcells,
                 vic_out_variables,
                 eval_file_names,
                 eval_var_names,
                 eval_path,
                    cal_multi):

        self.datastart = datetime.datetime.strptime(cal_start, '%Y-%m-%d')  # calibration start
        self.dataend = datetime.datetime.strptime(cal_end, '%Y-%m-%d')  # calibration end


        self.cal_multi = cal_multi
        self.gridcells = dfgridcells
        self.vic_out_variables = vic_out_variables
        self.eval_file_names = eval_file_names
        self.eval_dataset_number = len(self.eval_file_names)
        self.eval_var_names = eval_var_names
        self.eval_path = eval_path


        # model parameters to calibrate
        self.cal_mode = cal_mode
        self.params = params
        self.dir_work = dir_work
        self.dir_scripts = dir_scripts

        #self.RiverReachName = [i for i in river_reachid.keys()]
        self.time_step = time_step                                     # monthly/daily



        self.obj_funct = obj_funct_obj
        self.obj_f_opt_direction = obj_f_opt_direction

        self.obj_funct_param_flag = obj_funct_param_flag
        self.obj_funct_param = obj_funct_param

        self.curdir = os.getcwd()

        #print("current directory is {}".format(self.curdir))

        # below handling NaN when datasets have gaps.
        self.counter = 0
        self.n_gridcells = len(self.gridcells)
        self.flagnan = False

        if self.cal_mode == "mpi":
            self.call = str(int(os.environ['OMPI_COMM_WORLD_RANK']) + 2)
        else:
            pass
        return



    def parameters(self):

        self.par = spotpy.parameter.generate(self.params)

        return self.par

    def simulation(self, vector):

        if self.cal_mode == "seq":

            print("running sequential mode")

            globalFile = glob.glob(os.path.join(self.dir_work, "global*"))[0]
            paramFile = glob.glob(os.path.join(self.dir_work, "param*.nc"))[0]

            format_soil_params(paramFile, **{name.name: param for name, param in zip(self.params, vector)})

            subprocess.run(f'vic_image.exe -g {globalFile}', stderr = subprocess.PIPE,stdout = subprocess.PIPE, shell=True)

            # EXTRACT FLUXES
            print("open simulation")
            file = xr.open_dataset(glob.glob(self.dir_work + "/fluxes*.nc")[0], autoclose=True)
            sim = {}
            for ivar in self.vic_out_variables:
                if ivar == "OUT_SOIL_MOIST":
                    sim[ivar] = file[ivar][:, 0, ...] #get surface layer
                elif ivar == "OUT_EVAP":
                    sim[ivar] = file[ivar]

            date = pd.date_range(self.datastart, self.dataend)


            outer = {}
            for isim in sim:
                out = {}
                for gr in self.gridcells.index:
                    s = self.gridcells.loc[gr]
                    out[gr] = sim[isim].sel(lat=s.lat, lon=s.lon).values
                simdf = pd.DataFrame(out)
                simdf["time"] = date
                simdf = simdf.set_index("time")
                outer[isim] = simdf#.rolling(10).mean().dropna() ## testing rolling mean


            if self.time_step == "M":
                # monthly calibration
                simSeries = []
                for ix,iout in enumerate(outer):
                    simSeries.append([outer[iout].resample("M").agg(np.nanmean).iloc[:, i].values for i in range(len(self.gridcells))])

                print(f"length monthly observation time series {len(simSeries )}")

                if self.flagnan:
                    for i in range(self.eval_dataset_number):
                        simSeries[i] = [arr for i, arr in enumerate(simSeries[i]) if i not in self.index]


            if self.time_step == "D":
                # daily calibration
                simSeries = []
                for ix,iout in enumerate(outer):
                    simSeries.append([outer[iout].iloc[:, i].values for i in range(len(self.gridcells))])

                print(f"length daily observation time series {len(simSeries )}")


            if self.cal_multi:

                simSeries = simSeries[0] + simSeries[1]

            else:
                simSeries = simSeries[0]

            return simSeries

        else:

            #print(" ***** i am RUNNING SIMULATION in process rank: {}".format(self.call))
            # And generate a new folder with all underlying files
            self.parall_dir = self.dir_work + "/run_" + self.call  # call
            copytree(self.dir_work + "/run", self.parall_dir, ignore=ignore_patterns('tests', '*.py', 'fluxes*'))
            os.chdir(self.parall_dir)
            # rename files
            #print("rename global")
            globalFile_old = glob.glob("global*.txt")[0]
            globalFile_new = globalFile_old.split(".")[0] + "_{}.txt".format(self.call)

            # print("rename param")
            paramFile_old = glob.glob("param*.nc")[0]
            paramFile_new = paramFile_old.split(".")[0] + "_{}.nc".format(self.call)
            os.rename(paramFile_old, paramFile_new)


            # edit global file
            # print("edit global..")
            edit_vic_global(file=globalFile_old, par_file=paramFile_new, parall_dir=self.parall_dir,
                            globalFile_new=globalFile_new)

            os.remove(globalFile_old)


            try:

                format_soil_params(paramFile_new,
                                   **{name.name: param for name, param in zip(self.params, vector)})  # param_table
            except:
                print("format soil params failed")

            print("executing VIC model...")

            try:

                for i in os.environ:
                    if "OMPI" in i or "PMIX" in i:
                        del os.environ[i]

                subprocess.check_output(f'vic_image.exe -g {globalFile_new}', shell=True)

            except:
                print("vic run failed")


            # EXTRACT FLUXES

            try:
                file = xr.open_dataset(glob.glob(self.parall_dir  + "/fluxes*.nc")[0], autoclose=True)
                sim = {}
                for ivar in self.vic_out_variables:
                    if ivar == "OUT_SOIL_MOIST":
                        sim[ivar] = file[ivar][:, 0, ...]  # get surface layer
                    elif ivar == "OUT_EVAP":
                        sim[ivar] = file[ivar]


            except:
                print("can t open sim file")

            date = pd.date_range(self.datastart, self.dataend)

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
                if self.time_step == "M":
                    # monthly calibration
                    simSeries = []
                    for ix,iout in enumerate(outer):
                        simSeries.append([outer[iout].resample("M").agg(np.nanmean).iloc[:, i].values for i in range(len(self.gridcells))])

                    if self.flagnan:
                        for i in range(self.eval_dataset_number):
                            simSeries[i] = [arr for i, arr in enumerate(simSeries[i]) if i not in self.index]

                    print(f"length monthly simulation time series {len(simSeries )}")

            except:
                print("bordello con la version monthly")

            if self.time_step == "D":
                # daily calibration
                simSeries = []
                for ix,iout in enumerate(outer):
                    simSeries.append([outer[iout].iloc[:, i].values for i in range(len(self.gridcells))])

                print(f"length daily simulation time series {len(simSeries )}")


            if self.cal_multi:
                simSeries = simSeries[0] + simSeries[1]
            else:
                simSeries = simSeries[0]

            #print(simSeries)
            os.chdir(self.curdir)

            remove_tree(self.parall_dir)
            #rmtree(self.parall_dir, ignore_errors=True)
            print("removing folders")

            print("simulation length")
            print(len(simSeries))

            return simSeries



            #print(" ----- finished simulation in process rank: {} simulation file produced?: {}".format(self.call,simSeries is not None))



    def evaluation(self):

        print("opening evaluation")


        eval = {}
        for i,file in enumerate(self.eval_file_names):
            eval[file] = xr.open_dataset(self.eval_path + "/" + file,autoclose=True)[self.eval_var_names[i]]
            eval[file]["time"] = eval[file].indexes['time'].normalize()
            eval[file] = eval[file].loc[self.datastart:self.dataend] # eval[file].loc[self.datastart:(self.dataend + datetime.timedelta(days=1))]
        print("OK")


        date = pd.date_range(self.datastart, self.dataend)

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

        if self.time_step == "M":
            # monthly calibration



            print("COUNTER:")
            print(self.counter)

            obsSeries = []
            for ix, iout in enumerate(outer):
                obsSeries.append([outer[iout].resample("M").agg(np.nanmean).iloc[:, i].values for i in range(len(self.gridcells))])

            print(f"length monthly observation time series {len(obsSeries )}")

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



                for i in range(self.eval_dataset_number):
                    obsSeries[i] = [arr for i,arr in enumerate(obsSeries[i]) if i not in self.index]

            if self.flagnan:
                self.n_gridcells = len(self.gridcells.iloc[self.keepindex, :])

                pickle_out = open(f"{self.dir_work}/gridcell_index.pickle","wb")

                pickle.dump(self.keepindex, pickle_out)
                pickle_out.close()

            elif self.counter >=1 and self.flagnan:
                obsSeries[0] = obsSeries[0][self.index]

                print("removing index nan when counter >1------------------------")
                print(self.index)

            self.counter = self.counter + 1

            #print(obsSeries)


        if self.time_step == "D":
            # daily calibration
            obsSeries = []
            for ix, iout in enumerate(outer):
                obsSeries.append([outer[iout].iloc[:, i].values for i in range(len(self.gridcells))])

        ###

        if self.cal_multi:
            obsSeries = obsSeries[0] + obsSeries[1]
        else:
            obsSeries = obsSeries[0]

        print("observation length")
        print(len(obsSeries))

        return obsSeries


    def objectivefunction(self, simulation, evaluation,params=None):
        """
        :param simulation: simulation time series
        :param evaluation: observation time series
        :return:
        """

        print("obj function")
        if self.cal_multi:
            print("is multi")
            # TODO: here code to handle more than two datasets
            simulation = [simulation[:self.n_gridcells],simulation[self.n_gridcells:]]
            evaluation = [evaluation[:self.n_gridcells],evaluation[self.n_gridcells:]]
        else:
            pass

        self.weights = [1]*self.n_gridcells

        print("obj function")

        if self.cal_mode == "mpi":

            if self.cal_multi:  # multi-obj calibration
                objectivefunction = []
                for ix,(eval,sim) in enumerate(zip(evaluation,simulation)):

                    if self.obj_funct_param_flag:
                        objectivefunction.append([self.obj_funct(eval[i], sim[i],**self.obj_funct_param) for i in range(self.n_gridcells)])
                    else:
                        objectivefunction.append([self.obj_funct(eval[i], sim[i]) for i in range(self.n_gridcells)])

                    if self.obj_f_opt_direction == "min":
                        objectivefunction[ix] = float(np.sum(np.asarray([w * objectivefunction[ix][i] for i, w in enumerate(self.weights)])) / np.sum(self.weights))
                    elif self.obj_f_opt_direction == "max":
                        objectivefunction[ix] =  - float(
                            np.sum(np.asarray([w * objectivefunction[ix][i] for i, w in enumerate(self.weights)])) / np.sum(self.weights))
                    else:
                        print("warning: either min or max")

                return float(np.mean(np.array(objectivefunction)))

            else: # non multi

                if self.obj_funct_param_flag:
                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i], **self.obj_funct_param) for i in range(self.n_gridcells)]
                else:
                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i]) for i in range(self.n_gridcells)]

                if self.obj_f_opt_direction == "min":
                    objectivefunction = float(
                        np.sum(np.asarray([w * objectivefunction[i] for i, w in enumerate(self.weights)])) / np.sum(
                            self.weights))
                elif self.obj_f_opt_direction == "max":
                    objectivefunction = - float(
                        np.sum(np.asarray([w * objectivefunction[i] for i, w in enumerate(self.weights)])) / np.sum(
                            self.weights))

                return objectivefunction
        else:
            if self.cal_multi:  # multi-obj calibration
                objectivefunction = []
                for ix,(eval,sim) in enumerate(zip(evaluation,simulation)):

                    if self.obj_funct_param_flag:
                        objectivefunction.append([self.obj_funct(eval[i], sim[i],**self.obj_funct_param) for i in range(self.n_gridcells)])
                    else:
                        objectivefunction.append([self.obj_funct(eval[i], sim[i]) for i in range(self.n_gridcells)])

                    if self.obj_f_opt_direction == "min":
                        objectivefunction[ix] = float(np.sum(np.asarray([w * objectivefunction[ix][i] for i, w in enumerate(self.weights)])) / np.sum(self.weights))
                    elif self.obj_f_opt_direction == "max":
                        objectivefunction[ix] =  - float(
                            np.sum(np.asarray([w * objectivefunction[ix][i] for i, w in enumerate(self.weights)])) / np.sum(self.weights))
                    else:
                        print("warning: either min or max")

                return float(np.mean(np.array(objectivefunction)))

            else: # non multi
                if self.obj_funct_param_flag:
                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i], **self.obj_funct_param) for i in range(self.n_gridcells)]
                else:
                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i]) for i in range(self.n_gridcells)]

                if self.obj_f_opt_direction == "min":
                    objectivefunction = float(
                        np.sum(np.asarray([w * objectivefunction[i] for i, w in enumerate(self.weights)])) / np.sum(
                            self.weights))
                elif self.obj_f_opt_direction == "max":
                    objectivefunction = - float(
                        np.sum(np.asarray([w * objectivefunction[i] for i, w in enumerate(self.weights)])) / np.sum(
                            self.weights))


                return objectivefunction


