# --- module import
import os,sys
sys.path = ["/home/iff/research/spotpy"] + sys.path

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
from cal_spotpy_functions import read_clusters,set_interaction,pickling,_parseConfig,_readFromFile,create_sampling_gridcells

from spotpy_setup_sea import spotpy_setup



# --- define main function that manage calibration

def calibrate(cal_mode,
              dir_work,
              dir_scripts,
              cal_start,
              cal_end,
              time_step,
              params,
              cal_output_name,
              cal_alg_obj,
              obj_funct_obj,
              obj_f_opt_direction,
              cal_alg_param_flag,
              cal_alg_param,
              obj_funct_param_flag,
              obj_funct_param,
              param_file,
              vic_out_variables,
              eval_file_names,
              eval_var_names,
              eval_path,
              cal_multi,
              rep):



    # create dataframe with grid cells to extract evaluation

    dfgridcells = create_sampling_gridcells(param_file)

    cal_setup = spotpy_setup(dir_work,
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
                             cal_multi
                             )


    outCal = os.path.join(dir_work, cal_output_name)
    if cal_multi:
        sampler = cal_alg_obj(cal_setup, dbname=outCal, dbformat='csv', parallel=cal_mode,save_sim=True)
    else:
        sampler = cal_alg_obj(cal_setup, dbname=outCal, dbformat='csv', parallel=cal_mode,save_sim=True)



    results = []
    if cal_alg_param_flag:
        sampler.sample(rep, **cal_alg_param)
    else:
        sampler.sample(rep)



    results.append(sampler.getdata())

    return  cal_setup,sampler,results




if __name__ == "__main__":

    # option lumped/distributed

    config_global = _readFromFile(sys.argv[1])#sys.argv[1])#) #sys.argv[1])


    global_options = _parseConfig(config_global)




    cal_mode = global_options['vic_config']['cal_mode']
    cal_alg = global_options['vic_config']['cal_alg']

    obj_funct = global_options['vic_config']['obj_funct']
    obj_f_opt_direction = global_options['vic_config']['obj_f_opt_direction']

    cal_obj_multi = eval(global_options['vic_config']['cal_obj_multi'])

    vic_out_variables = [i for i in global_options['vic_config']['vic_out_variables'].split(",")]
    eval_file_names = [i for i in global_options['vic_config']['eval_file_names'].split(",")]
    eval_var_names = [i for i in global_options['vic_config']['eval_var_names'].split(",")]
    eval_path = global_options['vic_config']['path_eval']

    dir_work = global_options['vic_config']['dir_work']
    dir_scripts = global_options['vic_config']['dir_scripts']
    param_file = global_options['vic_config']['param_file']

    cal_multi = eval(global_options['vic_config']['cal_obj_multi'])

    cal_start = global_options['vic_config']['cal_start']  # starting sim time
    cal_end = global_options['vic_config']['cal_end']  # end sim time
    time_step = global_options['vic_config']['time_step']  # calibration timestep (D= daily, M=monthly)
    n_of_runs = int(global_options['vic_config']['n_of_runs'])  # number of runs

    cal_output_name = global_options['vic_config']['cal_output_name']


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

    cal_setup, sampler, results = calibrate(cal_mode,
                                            dir_work,
                                            dir_scripts,
                                            cal_start,
                                            cal_end,
                                            time_step,
                                            params,
                                            cal_output_name,
                                            cal_alg_obj,
                                            obj_funct_obj,
                                            obj_f_opt_direction,
                                            cal_alg_param_flag,
                                            cal_alg_param,
                                            obj_funct_param_flag,
                                            obj_funct_param,
                                            param_file,
                                            vic_out_variables,
                                            eval_file_names,
                                            eval_var_names,
                                            eval_path,cal_multi,
                                            rep=n_of_runs)




