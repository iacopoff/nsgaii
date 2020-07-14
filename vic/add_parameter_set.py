
import netCDF4 as nc
import numpy as np
import sys
import pickle

def format_soil_params_distributed(nc_file,gridcells, **kwargs):
    param_list = [i for i in kwargs]
    param_values = [kwargs[i] for i in kwargs]
    param_cal = nc.Dataset(nc_file, 'r+')
    param_cal.set_auto_mask(False)


    grid_array = param_cal.variables['gridcell'][:]

    for i, param in enumerate(param_list):
        if "1d" in param:
            p = param.split(".")[0]
            p2 = param.split("1d")[0]
            gr = gridcells[p][param]
            # got index from param cluster
            results = []
            cells = []
            for s in gr:
                idx = np.argwhere((grid_array - s) == 0)
                if len(idx) > 0:
                    results.append(idx[0])
                    cells.append(s)


            #a1 = pd.DataFrame({"Ds1d.3":grd[tuple(np.array(results).T)]}) #.to_csv(r"\\hydra\mas1261$\wp3\VIC\ba_bsn_025\run_190724_distr\idcels.csv")
            #a1.to_csv(r"\\hydra\mas1261$\wp3\VIC\ba_bsn_025\run_190724_distr\babasin.csv",index=False)
            #from matplotlib import pyplot as plt
            var = param_cal.variables[p2][:]

            #grd = param_cal.variables["gridcell"][:]


            var[tuple(np.array(results).T)]  = param_values[i]
            #plt.imshow(var[::-1])
            #plt.imshow(grid_array)
            param_cal.variables[p2][:] = var


        elif "2d" in param:
            p = param.split(".")[0]
            p2 = param.split("2d")[0]
            gr = gridcells[p][param]
            # got index from param cluster
            results = []
            cells = []
            for s in gr:
                idx = np.argwhere((grid_array - s) == 0)
                if len(idx) > 0:
                    results.append(idx[0])
                    cells.append(s)


            var = param_cal.variables[p2][:]
            var[1][tuple(np.array(results).T)] = param_values[i]

            param_cal.variables[p2][:] = var


        elif "3d" in param:
            p = param.split(".")[0]
            p2 = param.split("3d")[0]
            gr = gridcells[p][param]
            # got index from param cluster
            results = []
            for s in gr:
                idx = np.argwhere((grid_array - s) == 0)
                if len(idx) > 0:
                    results.append(idx[0])


            var = param_cal.variables[p2][:]
            var[2][tuple(np.array(results).T)]  = param_values[i]

            param_cal.variables[p2][:] = var


    param_cal.close()


    return



def read_par(par_cells,bestparset):
    cell_pickle = open(par_cells, "rb")
    par_pickle = open(bestparset, "rb")
    cell_pickle = pickle.load(cell_pickle )
    par_pickle = pickle.load(par_pickle)

    return par_pickle,cell_pickle



def main(nc_file,par_cells,bestparset):

    parameters, gridcells= read_par(par_cells,bestparset)

    format_soil_params_distributed(nc_file,gridcells,**parameters)

    return 0

if __name__ == "__main__":

    nc_file =r"/projects/mas1261/wp3/VIC/ca_025/test/VIC_parameter_vtm_025.nc"
    bestparset = r"/projects/mas1261/wp3/VIC/ca_025/test/bestcalparset_cal_distr_kge.pickle"
    par_cells = "/projects/mas1261/wp3/VIC/ca_025/test/par_cells.pickle"

    arg = sys.argv[1:]

    main(nc_file = arg[0],
         par_cells= arg[1],
         bestparset = arg[2])
