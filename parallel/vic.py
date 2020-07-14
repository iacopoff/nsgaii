import os,sys

sys.path = ["/home/iff/research/dev/vic_calibration"] + sys.path

import numpy as np
import time
from pathlib import Path
from shutil import rmtree
import numpy as np
from edit_config import edit_vic_global,edit_routing_config_dev

from dask.distributed import get_worker
from shutil import copytree,ignore_patterns
import subprocess
from format_soil_params import format_soil_params_distributed,format_soil_params
from run_routing import *
from cal_spotpy_functions import read_clusters,set_interaction,pickling,_parseConfig,_readFromFile
import glob


class VIC:

    def __init__(self,parentDir,n_var,n_obj):
        self.parentDir = parentDir
        self.n_var = n_var
        self.n_obj = n_obj


    def evaluate(self,x):
        print(f"WORKER ID IS: {get_worker().id}")

        workerID = "_".join(get_worker().id.split("-")[:2])
        workerDir = os.path.join(self.parentDir,workerID)

        # create directory
        #Path(workerDir).mkdir(parents=True, exist_ok=True)

        copytree(os.path.join(self.parentDir,"run"),workerDir,ignore = ignore_patterns('tests','*py','fluxes*'))

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

        print("executing VIC model...")

        try:

            subprocess.check_output(f'vic_image.exe -g {globalFile_new}', shell=True)

        except:
            print("vic run failed")



        # EXTRACT FLUXES

        sim = self.simulation(x)
        os.chdir(self.parentDir)
        rmtree(workerDir)

        res = np.random.random((self.n_var,self.n_obj))

        return res


    def simulation(x):
        return x


    def evaluation(x):
        return

    def objective_functions(simulation,evaluation):
        return 
