# import netCDF4 as nc
# import numpy as np
# import sys
# import os
# import pandas as pd


from run_routing import *
from format_soil_params import SEA_format_soil_params


# user config ----

param_path = '/projects/fws1273/wp3/calibration/results/selected_params.csv'

# VIC parameter file
orig_vic_param_path = '/projects/fws1273/wp3/calibration/results/VIC_parameter_025.nc'

# cluster file cells ids
cluster_id_cells_path = '/projects/fws1273/wp3/calibration/results/SEA_cluster_masked_cluster_idn_clusters4.csv'

n_params = 5



# add selected calibrated parameter to VIC parameter file. Each cluster receives a set of parameters.

cluster_ids = pd.read_csv(cluster_id_cells_path)

param_table = pd.read_csv(param_path,dtype={'run_index':'int'})

for cluster in param_table.cluster:

    print(f"replace parameters for cluster: {cluster}")

    # select cell ids for a cluster
    gridcell_ids = cluster_ids.loc[cluster_ids.cluster == cluster,"id"]

    # get all the tables
    run_index = param_table.loc[param_table.cluster == cluster,"run_index"].values[0]
    run_table_path = param_table.loc[param_table.cluster == cluster,"run"].values[0]
    config_file_path = param_table.loc[param_table.cluster == cluster,"config_file"].values[0]

    # get parameter values and names
    params_dict = pd.read_table(run_table_path,sep=",").iloc[run_index,1:(n_params+1)].to_dict()

    print(f"parameters to replace: {params_dict}")

    # replace parameter in VIC parameter file
    SEA_format_soil_params(orig_vic_param_path,gridcell_ids,**params_dict)
