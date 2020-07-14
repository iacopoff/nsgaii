

# --- module import
import os,sys
sys.path = ["/projects/mas1261/wa_software/spotpy1.5.6"] + sys.path

import spotpy
import datetime
import subprocess

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
    def __init__(self,model_type,dir_work,dir_routexe,dir_scripts,river_reachid,cal_start,cal_end,time_step,params,clusters,
                 relation,interactions,weights,discharge_summary,discharge_folder,cal_mode,obj_funct_obj,obj_f_opt_direction,
                 obj_funct_param_flag, obj_funct_param):

        self.datastart = datetime.datetime.strptime(cal_start, '%Y-%m-%d')  # calibration start
        self.dataend = datetime.datetime.strptime(cal_end, '%Y-%m-%d')  # calibration end


        # model parameters to calibrate
        self.cal_mode = cal_mode
        self.params = params
        self.discharge_summary = discharge_summary
        self.discharge_folder=discharge_folder

        self.dir_work = dir_work
        self.dir_routexe = dir_routexe
        self.dir_scripts = dir_scripts
        self.river_reachid = river_reachid   # loop for multi site calibration
        #self.RiverReachName = [i for i in river_reachid.keys()]
        self.time_step = time_step                                     # monthly/daily
        self.weights = np.asarray(weights)                           # weights for multi site calibration
        self.n_streams = len(self.river_reachid)

        # for generating parameters
        self.clusters = clusters
        self.relation = relation
        self.interactions = interactions

        self.model_type = model_type
        self.obj_funct = obj_funct_obj
        self.obj_f_opt_direction = obj_f_opt_direction
        self.routing_config_name = "route.control"

        self.obj_funct_param_flag = obj_funct_param_flag
        self.obj_funct_param = obj_funct_param

        self.curdir = os.getcwd()
        #print("current directory is {}".format(self.curdir))
        if self.cal_mode == "mpi":
            self.call = str(int(os.environ['OMPI_COMM_WORLD_RANK']) + 2)
        else:
            pass
        return

    def parameters(self):
        if self.model_type == "distributed":
            self.par,self.parnames = set_interaction(self.params, self.clusters, self.relation,self.interactions)
        else:
            self.par = spotpy.parameter.generate(self.params)
        return self.par

    def simulation(self, vector):



        if self.cal_mode == "seq":
            print("running sequential mode")
            globalFile = glob.glob(os.path.join(self.dir_work, "global*"))[0]
            paramFile = glob.glob(os.path.join(self.dir_work, "param*.nc"))[0]

            # create soil parameter file with new set of variables
            print("formatting parameter file...")
            if self.model_type == "distributed":
                print("running distributed mode")
                format_soil_params_distributed(nc_file =paramFile,gridcells=self.clusters,**{name:param for name, param in zip(self.parnames, vector)})  # param_table
            else:
                print("running lumped mode")
                format_soil_params(paramFile, **{name.name: param for name, param in zip(self.params, vector)})
            print("...formatting done!")


            print("executing vic...")
            subprocess.run(f'vic_image.exe -g {globalFile}', stderr = subprocess.PIPE,stdout = subprocess.PIPE, shell=True)


            print("runnig routing")
            run_routing_dev(VIC_folder= self.dir_work, routing_f=self.dir_routexe, SEGID=self.river_reachid,
                        startTime=pd.Timestamp(self.datastart).strftime(format="%Y-%m-%d"),
                        endTime=pd.Timestamp(self.dataend).strftime(format="%Y-%m-%d"),
                        routing_method=['IRFroutedRunoff'])
            print("...succesful!")


            simCsv = pd.read_csv(os.path.join( self.dir_work, "sim_discharge.csv"), index_col=0,parse_dates=True)

            simCsv.index.name ="time"


            if self.time_step == "M":
                # monthly calibration
                simCsv.columns = [i for i in self.river_reachid]
                simSeries = simCsv.resample("M").sum().iloc[:,:self.n_streams]    #values[:,:self.n_streams]
                print(f"length monthly observation time series {len(simSeries )}")
            if self.time_step == "D":
                # daily calibration
                simCsv.columns = [i for i in self.river_reachid]
                simSeries = simCsv.iloc[:,:self.n_streams]   #values[:,:self.n_streams]
                print(f"length daily observation time series {len(simSeries )}")

            if self.n_streams >1: # multi-sites calibration
                simSeries = [simSeries.iloc[:, i].values for i in range(self.n_streams)]
            else:
                simSeries = simSeries.values[:,0]

            return simSeries

        else:

            #print(" ***** i am RUNNING SIMULATION in process rank: {}".format(self.call))
            # And generate a new folder with all underlying files
            self.parall_dir = self.dir_work + "/run_" + self.call  # call
            copytree(self.dir_work + "/run", self.parall_dir, ignore=ignore_patterns('tests', '*.py', 'fluxes*'))
            # print("created new run directory {}".format(self.parall_dir))
            self.routing_dir = self.dir_routexe + "_" + self.call  # call ########
            copytree(self.dir_routexe,self.routing_dir)

            os.chdir(self.parall_dir)
            # rename files
            #print("rename global")
            globalFile_old = glob.glob("global*.txt")[0]
            globalFile_new = globalFile_old.split(".")[0] + "_{}.txt".format(self.call)

            # print("rename param")
            paramFile_old = glob.glob("param*.nc")[0]
            paramFile_new = paramFile_old.split(".")[0] + "_{}.nc".format(self.call)
            os.rename(paramFile_old, paramFile_new)
            # print(glob.glob(f"param*2015*.nc")[0])

            # edit global file
            # print("edit global..")
            edit_vic_global(file=globalFile_old, par_file=paramFile_new, parall_dir=self.parall_dir,
                            globalFile_new=globalFile_new)

            os.remove(globalFile_old)
            # print(glob.glob( f"global*.txt")[0])

            # edit routing config
            config_file = os.path.join(self.dir_routexe, "settings", self.routing_config_name)

            # print(config_file)
            try:
                edit_routing_config_dev(config_file=config_file, parall_dir=self.routing_dir,
                                        config_new=self.routing_config_name)
            except:
                print("failed editing config")

                # create soil parameter file with new set of variables
            # print("formatting parameter file...")
            #
            try:

                if self.model_type == "distributed":
                    format_soil_params_distributed(nc_file=paramFile_new, gridcells=self.clusters,
                                                   **{name: param for name, param in
                                                      zip(self.parnames, vector)})  # param_table
                else:
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


            print("executing routing...")
            try:

                run_routing_dev(VIC_folder=self.parall_dir, routing_f=self.routing_dir, SEGID=self.river_reachid
                                , startTime=pd.Timestamp(self.datastart).strftime(format="%Y-%m-%d"),
                                endTime=pd.Timestamp(self.dataend).strftime(format="%Y-%m-%d"),
                                routing_method=['IRFroutedRunoff'])
            except:
                print("routing failed")

            simCsv = pd.read_csv(os.path.join(self.parall_dir, "sim_discharge.csv"), index_col=0,parse_dates=True)

            simCsv.index.name ="time"


            if self.time_step == "M":
                # monthly calibration
                simCsv.columns = [i for i in self.river_reachid]
                simSeries = simCsv.resample("M").sum().iloc[:,:self.n_streams]    #values[:,:self.n_streams]

                print(f"length monthly observation time series {len(simSeries )}")
            if self.time_step == "D":
                # daily calibration
                simCsv.columns = [i for i in self.river_reachid]
                simSeries = simCsv.iloc[:,:self.n_streams]   #values[:,:self.n_streams]
                print(f"length daily observation time series {len(simSeries )}")

            if self.n_streams >1: # multi-sites calibration

                simSeries = [simSeries.iloc[:, i].values for i in range(self.n_streams)]


                os.chdir(self.curdir)

                remove_tree(self.parall_dir)
                remove_tree(self.routing_dir)
                return simSeries
            else:
                simSeries = simSeries.values[:,0]

                os.chdir(self.curdir)

                remove_tree(self.parall_dir)
                remove_tree(self.routing_dir)
                return simSeries



            #print(" ----- finished simulation in process rank: {} simulation file produced?: {}".format(self.call,simSeries is not None))



    def evaluation(self):
        """
        :return: observations from the
        """
        df=pd.read_excel(self.discharge_summary)
        selection= df.loc[df['routing_id'].isin(self.river_reachid)]
        print(" calibrating with " + str(selection['station'].tolist()))


        #valFile = os.path.join(self.dir_work, "babasin_obs.csv")  # calibration # TODO: probably this could be moved into main
        obs_in=pd.DataFrame()
        for i in self.river_reachid:
            id=selection['routing_id']==i
            valFile=selection[id]['excel_name'].tolist()[0]
            obs_in_t = pd.read_csv(self.discharge_folder+"/"+valFile+".csv", usecols=[0,1])
            obs_in_t.columns = ["time", i]
            try:
                obs_in_t['time'] = pd.to_datetime(obs_in_t.time, format="%d/%m/%Y")
            except ValueError:
                obs_in_t['time'] = pd.to_datetime(obs_in_t.time, format="%Y-%m-%d")
            obs_in_t = obs_in_t[(obs_in_t.time >= pd.Timestamp(self.datastart)) & (obs_in_t.time <= pd.Timestamp(self.dataend))]
            obs_in_t=obs_in_t.set_index("time")
            obs_in[i]=obs_in_t[i]

        print("observed from excel")
        #print(obs_in)


        if self.time_step == "M":
            obsSeries = obs_in#.set_index("time")
            #obsSeries = obsSeries[[i for i in self.RiverReachName]]
            obsSeries = obsSeries.resample("M").sum()#.iloc[:,:self.n_streams]
            print(f"length monthly observation time series {len(obsSeries )}")
        if self.time_step == "D":
            obsSeries = obs_in #[[i for i in self.RiverReachName]].iloc[:,:self.n_streams]
            print(f"length daily observation time series {len(obsSeries )}")



        if self.n_streams > 1:  # multi-sites calibration

            obsSeries = [obsSeries.iloc[:, i].values for i in range(self.n_streams)]

            return obsSeries
        else:
            obsSeries = obsSeries.values[:, 0]

            return obsSeries







    def objectivefunction(self, simulation, evaluation,params=None):
        """
        :param simulation: simulation time series
        :param evaluation: observation time series
        :return:
        """

        print("obj function")

        if self.n_streams > 1:

            if self.cal_mode == "mpi":
                # multi site option

                if self.obj_funct_param_flag:
                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i],**self.obj_funct_param) for i in range(self.n_streams)]
                else:
                    print(evaluation)
                    print(simulation)

                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i]) for i in range(self.n_streams)]

                if self.obj_f_opt_direction == "min":
                    objectivefunction = float(np.sum(np.asarray([w*objectivefunction[i] for i,w in enumerate(self.weights)]))/np.sum(self.weights))
                elif self.obj_f_opt_direction == "max":
                    objectivefunction = -float(np.sum(np.asarray([w * objectivefunction[i] for i, w in enumerate(self.weights)])) / np.sum(self.weights))
                else:
                    print("warning: either min or max")

            else:

                if self.obj_funct_param_flag:
                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i],**self.obj_funct_param) for i in range(self.n_streams)]
                else:
                    objectivefunction = [self.obj_funct(evaluation[i], simulation[i]) for i in range(self.n_streams)]

                if self.obj_f_opt_direction == "min":
                    objectivefunction = float(
                        np.sum(np.asarray([w * objectivefunction[i] for i, w in enumerate(self.weights)])) / np.sum(
                            self.weights))
                elif self.obj_f_opt_direction == "max":
                    objectivefunction = - float(
                        np.sum(np.asarray([w * objectivefunction[i] for i, w in enumerate(self.weights)])) / np.sum(
                            self.weights))
                else:
                    print("warning: either min or max")
        else:

            if self.obj_f_opt_direction == "min":
                if self.obj_funct_param_flag:
                    objectivefunction = self.obj_funct(evaluation, simulation,**self.obj_funct_param)
                else:
                    objectivefunction = self.obj_funct(evaluation, simulation)
            elif self.obj_f_opt_direction == "max":
                if self.obj_funct_param_flag:
                    objectivefunction = - self.obj_funct(evaluation, simulation,**self.obj_funct_param)
                else:
                    objectivefunction = - self.obj_funct(evaluation, simulation)
            else:
                print("warning: either min or max")


        return objectivefunction



# --- define main function that manage calibration

def calibrate(model_type,cal_mode,dir_work,dir_routexe,dir_scripts,river_reachid,cal_start,
              cal_end,time_step,params,rank_in,
              weights,cal_output_name,discharge_summary,discharge_folder,cal_alg_obj,obj_funct_obj,obj_f_opt_direction,
              cal_alg_param_flag,cal_alg_param,obj_funct_param_flag,obj_funct_param,interactions,rep):

    if model_type == "distributed":

        clusters,parnames,relation = read_clusters(rank_in)

        pickling(file=clusters, outdir_path=  dir_work)

        cal_setup = spotpy_setup(model_type,dir_work,dir_routexe,dir_scripts,river_reachid,cal_start,cal_end,
                                time_step,params,clusters,relation,interactions,weights,discharge_summary,
                                 discharge_folder,cal_mode,obj_funct_obj,obj_f_opt_direction, obj_funct_param_flag, obj_funct_param)

        outCal = os.path.join(dir_work, cal_output_name )
        sampler = cal_alg_obj(cal_setup, dbname=outCal, dbformat='csv',parallel = cal_mode)


        results = []  # empty list to append iteration results
        if cal_alg_param_flag:
            sampler.sample(rep, **cal_alg_param)
        else:
            sampler.sample(rep)

        #TODO: add function to extract log file errror and remove logs
        results.append(sampler.getdata())

    else:

        clusters = None
        relation  = None
        cal_setup = spotpy_setup(model_type,dir_work, dir_routexe, dir_scripts, river_reachid, cal_start, cal_end,
                                 time_step, params, clusters, relation, interactions,weights,
                                 discharge_summary, discharge_folder,
                                 cal_mode,obj_funct_obj,obj_f_opt_direction, obj_funct_param_flag, obj_funct_param)
        outCal = os.path.join(dir_work, cal_output_name)
        sampler = cal_alg_obj(cal_setup, dbname=outCal, dbformat='csv', parallel=cal_mode)



        results = []
        if cal_alg_param_flag:
            sampler.sample(rep, **cal_alg_param)
        else:
            sampler.sample(rep)

        # TODO: add function to extract log file errror and remove logs

        results.append(sampler.getdata())

    return  cal_setup,sampler,results




if __name__ == "__main__":

    # option lumped/distributed

    config_global = _readFromFile(sys.argv[1])#) #sys.argv[1])


    global_options = _parseConfig(config_global)



    model_type = global_options['vic_config']['model_type']
    cal_mode = global_options['vic_config']['cal_mode']
    cal_alg = global_options['vic_config']['cal_alg']

    obj_funct = global_options['vic_config']['obj_funct']
    obj_f_opt_direction = global_options['vic_config']['obj_f_opt_direction']

    dir_work = global_options['vic_config']['dir_work']
    dir_routexe = global_options['vic_config']['dir_routexe']
    discharge_summary = global_options['vic_config']['discharge_summary']
    discharge_folder = global_options['vic_config']['discharge_folder']
    dir_scripts = global_options['vic_config']['dir_scripts']
    river_reachid = [int(i) for i in global_options['vic_config']['river_reachid'].split(",")]  # rivers and river ID as defined in mizourute
    #print(river_reachid)
    cal_start = global_options['vic_config']['cal_start']  # starting sim time
    cal_end = global_options['vic_config']['cal_end']  # end sim time
    time_step = global_options['vic_config']['time_step']  # calibration timestep (D= daily, M=monthly)
    n_of_runs = int(global_options['vic_config']['n_of_runs'])  # number of runs
    param_file = global_options['vic_config']['param_file'] #TODO dove lo legge
    weights = [int(i) for i in global_options['vic_config']['weights'].split(",")]
    cal_output_name = global_options['vic_config']['cal_output_name']
    clusters_ranks = global_options['vic_config']['clusters_ranks']
    interactions = eval(global_options['vic_config']['interactions'])

    params = []
    for par in config_global['vic_parameters']:
        p = config_global['vic_parameters'][par].split(",")
        minmax = [float(i) for i in p[1:]]
        param_distr = getattr(spotpy.parameter, p[0])
        params.append(param_distr(par, *minmax))


    cal_alg_obj = getattr(spotpy.algorithms, cal_alg)
    obj_funct_obj = getattr(spotpy.objectivefunctions, obj_funct)

    cal_alg_param_flag = eval(config_global['vic_calalg_parameter']['flag'])
    cal_alg_param = {}
    if cal_alg_param_flag:
        for a in config_global['vic_calalg_parameter']:
            if a != 'flag':
                cal_alg_param[a] = eval(config_global['vic_calalg_parameter'][a])
    else:
        cal_alg_param = None

    obj_funct_param_flag = eval(config_global['vic_objfunct_parameter']['flag'])
    obj_funct_param = {}
    if obj_funct_param_flag:
        for a in config_global['vic_objfunct_parameter']:
            if a != 'flag':
                obj_funct_param[a] = eval(config_global['vic_objfunct_parameter'][a])
    else:
        obj_funct_param = None

    cal_setup, sampler, results = calibrate(model_type, cal_mode, dir_work, dir_routexe, dir_scripts, river_reachid,
                                            cal_start, cal_end, time_step, params,
                                            clusters_ranks,
                                            weights, cal_output_name,
                                            discharge_summary, discharge_folder, cal_alg_obj,obj_funct_obj,obj_f_opt_direction,
                                            cal_alg_param_flag,cal_alg_param,
                                            obj_funct_param_flag,obj_funct_param,interactions,
                                            rep=n_of_runs)




