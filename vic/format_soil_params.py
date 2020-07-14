#opening VIC paramter file and changing thew soil paramter

import netCDF4 as nc
import numpy as np


def format_soil_params(nc_file, **kwargs):
    param_list = [i for i in kwargs]

    param_values = [kwargs[i] for i in kwargs]

    param_cal = nc.Dataset(nc_file, 'r+')
    param_cal.set_auto_mask(False)
    grid_array = param_cal.variables['gridcell'][:]
    nodata = param_cal.variables['gridcell']._FillValue
    idx = np.vstack(np.argwhere(grid_array > nodata))
    for i, param in enumerate(param_list):

        if "1d" in param:
            p2 = param.split("1d")[0]
            var = param_cal.variables[p2][:]
            if len(var.shape) >= 3:
                var[0][tuple(idx.T)]= param_values[i]
            else:
                var[tuple(idx.T)] = param_values[i]
            #var[0][[*idx.T]]

            param_cal.variables[p2][:] = var
        elif "2d" in param:
            p2 = param.split("2d")[0]
            var = param_cal.variables[p2][:]
            #var[1][[*idx.T]] = param_values[i]
            var[1][tuple(idx.T)] = param_values[i]
            param_cal.variables[p2][:] = var
        elif "3d" in param:
            p2 = param.split("3d")[0]
            var = param_cal.variables[p2][:]
            #var[2][[*idx.T]] = param_values[i]
            var[2][tuple(idx.T)] = param_values[i]
            param_cal.variables[p2][:] = var

    # DEBUG
    # print(f"the parameter values in the netcdf are now: {var[2][[*idx.T]]}")
    # print("..and write netcdf to disk..")
    param_cal.close()

    # # debug, reopen
    # print("reopen netcdf to see if changes were saved:")
    # param_cal = nc.Dataset(nc_file, 'r+')
    # param_cal.set_auto_mask(False)
    # var = param_cal.variables["Ksat"][:]
    # print(f"Parameter value is now: {var[2][[*idx.T]]}")
    # print("close..")
    # param_cal.close()

    return




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



def SEA_format_soil_params(nc_file,gridcells, **kwargs):
    param_list = [i for i in kwargs]
    param_values = [kwargs[i] for i in kwargs]
    param_cal = nc.Dataset(nc_file, 'r+')
    param_cal.set_auto_mask(False)


    grid_array = param_cal.variables['gridcell'][:]

    for i, param in enumerate(param_list):
        if "1d" in param:
            p2 = param.split("1d")[0][3:]
            gr = gridcells.values
            # got index from param cluster
            results = []
            cells = []
            for s in gr:
                idx = np.argwhere((grid_array - s) == 0)
                if len(idx) > 0:
                    results.append(idx[0])
                    cells.append(s)

            var = param_cal.variables[p2][:]
            var[tuple(np.array(results).T)]  = param_values[i]
            param_cal.variables[p2][:] = var


        elif "2d" in param:

            p2 = param.split("2d")[0][3:]
            gr = gridcells.values
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
            p2 = param.split("3d")[0][3:]
            gr = gridcells.values
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


def NSGAII_no_format_soil_params(nc_file,gridcells, param_names,param_values):
    param_cal = nc.Dataset(nc_file, 'r+')
    param_cal.set_auto_mask(False)

    import pdb; pdb.set_trace()

    grid_array = param_cal.variables['gridcell'][:]

    for i, param in enumerate(param_names):
        if "1d" in param:
            p2 = param.split("1d")[0]
            gr = gridcells.values
            # got index from param cluster
            results = []
            cells = []
            for s in gr:
                idx = np.argwhere((grid_array - s) == 0)
                if len(idx) > 0:
                    results.append(idx[0])
                    cells.append(s)

            var = param_cal.variables[p2][:]
            var[tuple(np.array(results).T)]  = param_values[i]
            param_cal.variables[p2][:] = var


        elif "2d" in param:

            p2 = param.split("2d")[0]
            gr = gridcells.values
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
            p2 = param.split("3d")[0]
            gr = gridcells.values
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


def NSGAII_format_soil_params(nc_file,param_names,param_values):
    param_cal = nc.Dataset(nc_file, 'r+')
    param_cal.set_auto_mask(False)

    #import pdb; pdb.set_trace()


    grid_array = param_cal.variables['gridcell'][:]
    nodata = param_cal.variables['gridcell']._FillValue
    idx = np.vstack(np.argwhere(grid_array > nodata))
    for i, param in enumerate(param_names):

        if "1d" in param:
            p2 = param.split("1d")[0]
            var = param_cal.variables[p2][:]
            if len(var.shape) >= 3:
                var[0][tuple(idx.T)]= param_values[i]
            else:
                var[tuple(idx.T)] = param_values[i]
            #var[0][[*idx.T]]

            param_cal.variables[p2][:] = var
        elif "2d" in param:
            p2 = param.split("2d")[0]
            var = param_cal.variables[p2][:]
            #var[1][[*idx.T]] = param_values[i]
            var[1][tuple(idx.T)] = param_values[i]
            param_cal.variables[p2][:] = var
        elif "3d" in param:
            p2 = param.split("3d")[0]
            var = param_cal.variables[p2][:]
            #var[2][[*idx.T]] = param_values[i]
            var[2][tuple(idx.T)] = param_values[i]
            param_cal.variables[p2][:] = var

    # DEBUG
    # print(f"the parameter values in the netcdf are now: {var[2][[*idx.T]]}")
    # print("..and write netcdf to disk..")
    param_cal.close()

    # # debug, reopen
    # print("reopen netcdf to see if changes were saved:")
    # param_cal = nc.Dataset(nc_file, 'r+')
    # param_cal.set_auto_mask(False)
    # var = param_cal.variables["Ksat"][:]
    # print(f"Parameter value is now: {var[2][[*idx.T]]}")
    # print("close..")
    # param_cal.close()

    return

