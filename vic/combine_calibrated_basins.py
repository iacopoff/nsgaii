import netCDF4 as nc
import numpy as np
import sys
import spotpy
from shutil import copyfile
import os
script_dir = "/projects/home/fmo/dmossgit/wa_data_analysis_branch/research"
sys.path.append(f"{script_dir}")
import pandas as pd
from shutil import copyfile

from vic_calibration.format_soil_params import format_soil_params_distributed, format_soil_params
from vic_calibration.run_routing import *

from vic_calibration.eval_functions import *
from vic_calibration.cal_spotpy_functions import read_clusters, set_interaction, pickling, _parseConfig, _readFromFile

#####

param_or="/projects/mas1261/wp3/VIC/calibration_output/param_template/param_cal_or.nc"
wk_copy="/projects/mas1261/wp3/VIC/calibration_output/param_"

val_sum=pd.read_excel("/projects/mas1261/wp3/VIC/calibration_output/validation_summary.xlsx",dtype="str")
#val_sum=val_sum.sort_values(by="area", ascending=True)
config_files=val_sum['config_file']
#we take the first best run for now
n_run=0

#creating parameter file for all files
for i,c in enumerate(config_files):
    param_file=wk_copy+val_sum.basin[i]+".nc"
    copyfile(param_or, param_file)
    config_global = _readFromFile(c)  # _readFromFile(sys.argv[1]) #'_readFromFile("/projects/mas1261/wp3/VIC/machu_025/run/191001/config.ini")  #  #  _readFromFile("/projects/mas1261/wp3/VIC/machu_025/run/191001/config.ini") # #)
    global_options = _parseConfig(config_global)
    run_id=val_sum.run_selected_river_flow[i].split("/")[n_run]
    clusters_ranks = global_options['vic_config']['clusters_ranks']
    model_type = global_options['vic_config']['model_type']
    cal_output_name = os.path.join(os.path.dirname(c),global_options['vic_config']['cal_output_name'])+".csv"
    if model_type=="lumped":
        params = []
        for par in config_global['vic_parameters']:
            p = config_global['vic_parameters'][par].split(",")
            minmax = [float(i) for i in p[1:]]
            param_distr = getattr(spotpy.parameter, p[0])
            params.append(param_distr(par, *minmax))
        n_par = len(params)

    else:
        params = []
        for par in config_global['vic_parameters']:
            p = config_global['vic_parameters'][par].split(",")
            minmax = [float(i) for i in p[1:]]
            param_distr = getattr(spotpy.parameter, p[0])
            params.append(param_distr(par, minmax[0], minmax[1]))
        clusters, parnames_initial, relation = read_clusters(clusters_ranks)

        flag = False
        for k in clusters.copy():
            for k2 in clusters[k].copy():
                if len(clusters[k][k2]) == 0:
                    clusters[k].pop(k2)
                    flag = True

        if flag:
            parnames = [i + ".1" for i in clusters.keys()]
        else:
            param, parnames = set_interaction(params, clusters, relation, interactions=False)

      #  param, parnames = set_interaction(params, clusters, relation,interactions=False)
        n_par = len(parnames)
    cal_file = pd.read_csv(cal_output_name)
    vector = cal_file.sort_values(by="like1", ascending=True).iloc[int(run_id), 1:n_par + 1]
    vector_list = vector.tolist()
    #get index of vectors
    print("getting run "+run_id+" of " +val_sum.basin[i])
    param_names = cal_file.columns[1:n_par + 1].tolist()
    if model_type=="lumped":
        format_soil_params(param_file, **{name.name: param for name, param in zip(params, vector)})
        par_nam=[x.replace("par", "") for x in param_names]
        param_df = pd.DataFrame(columns=par_nam, index=[1])
        for p in par_nam:
            vect_id = "par" + p
            param_df.loc[1, p] = vector[vect_id]
        param_df = param_df.astype(float).round(4)

    else:
        clusters_id = list(set([x.split(".")[1] for x in param_names]))
        par_nam = list(set([x.split(".")[0] for x in parnames]))
        param_df = pd.DataFrame(columns=par_nam, index=clusters_id)
        for cl in clusters_id:
            for p in par_nam:
                vect_id = "par" + p + "." + cl
                param_df.loc[cl, p] = vector[vect_id]
        param_df = param_df.astype(float).round(4)
        clusters, parnames_initial, relation = read_clusters(clusters_ranks)
        format_soil_params_distributed(nc_file=param_file, gridcells=clusters,
                                       **{name: param for name, param in zip(parnames, vector_list)})

    print("done")

##combining all parameters in one parameter

val_sum_sort=val_sum.sort_values(by="area_km2", ascending=True).reset_index(drop=True)
len_basins=len(val_sum_sort.basin)
netcdf_ON_TOP_file=wk_copy+"_merged.nc"
copyfile(wk_copy+val_sum_sort.basin[0]+".nc", netcdf_ON_TOP_file)
#variables to sobstitute
var_list=["depth","Ds","Dsmax","infilt","Ws"]
for i,basin in enumerate(val_sum_sort.basin):
    if i==len_basins-1:break
    on_top_ds = nc.Dataset(netcdf_ON_TOP_file,'r+')
    on_top_ds.set_auto_mask(False)
    below_ds = nc.Dataset(wk_copy+val_sum_sort.basin[i+1]+".nc")
    below_ds.set_auto_mask(False)
    for var in var_list:
        on_top_array = on_top_ds.variables[var][:]
        below_array = below_ds.variables[var][:]
        mask = (on_top_array == 0)
        new_array=np.copy(on_top_array)
        new_array[mask]=below_array[mask]
        on_top_array=new_array
        on_top_ds.variables[var][:]=on_top_array

    on_top_ds.close()



