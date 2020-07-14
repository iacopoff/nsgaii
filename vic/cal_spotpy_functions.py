import numpy as np
import pandas as pd
import spotpy
import pickle
import os
import configparser
import sys
import logging
import netCDF4 as nc


# --- functions for vic_cal_spotpy_distributed.py

def aggregate(multistream,flow_names):
    tot = multistream.sum(axis=1)
    scale = multistream[flow_names].div(tot,axis=0)
    #scale.apply(lambda x: x/tot,axis=0)
    return tot.values,scale


def read_clusters(rank_in):
    """
    :param:
        rank_in: path to the csv table holding parameter relations
    :return:
        par_dict: dictionary, keys are names of new generated cluster parameters and values are grid cell indeces
        parameters: numpy array with parameter names
        relation: the parameter relation among clusters
    """
    rank = pd.read_csv(rank_in)
    cluster_source = rank.cluster.values.astype("str")
    parameters = rank.parameter.values.astype("str")
    relation = rank.relation.values.astype("str")
    relation = [s.split(">") for s in relation]


    par_dict = {}
    for c,par in enumerate(parameters):
        temp = pd.read_csv(f"{os.path.join(os.path.dirname(rank_in),cluster_source[c])}")
        temp_rel = [ int(i) for i in relation[c]]
        par_dict2 = {}
        for i in temp_rel:
            par_dict2[".".join([par,str(i)])] = temp.loc[temp.cluster == i,"id"].values
        par_dict[par] =par_dict2

    return par_dict,parameters, relation



def set_interaction(params,clusters,relation,interactions = True):
    """

    :param params:
    :param clusters:
    :param relation:
    :return:
    """
    names = [n.name for n in params]
    parold = spotpy.parameter.generate(params)
    newpar = np.empty((len(names)*len(relation[0])),dtype=parold.dtype)
    sss = 0
    newnames = []
    if interactions:
        for ipar,par in enumerate(names):
            for iclust,clust in enumerate(relation[ipar]):
                ss =  ".".join([par,clust])
                if iclust == 0:
                    parold[ipar]["name"] = ss
                    newpar[sss] = parold[ipar]
                    newnames.append(parold[ipar]["name"])
                    sss += 1
                else:
                    fname = params[ipar].rndfunctype
                    func = getattr(spotpy.parameter, fname)
                    newpar[sss] =  spotpy.parameter.generate([func(ss,low=newpar[sss -1]["minbound"], high=newpar[sss -1]["random"])])
                    newnames.append(ss)
                    sss += 1
    else:
        for ipar,par in enumerate(names):
            for iclust,clust in enumerate(relation[ipar]):
                ss =  ".".join([par,clust])
                fname = params[ipar].rndfunctype
                func = getattr(spotpy.parameter, fname)
                # uniform function expect in this order: name, min, max, step, optguess but parold is step,optguess,min,max
                p = list(parold[ipar])[4:-1] + list(parold[ipar])[2:4]
                newpar[sss] = spotpy.parameter.generate([func(ss,*p)])
                newnames.append(ss)
                sss += 1

    return newpar,newnames




def pickling(file,outdir_path):

    pickle_out = open(f"{outdir_path}/par_cells.pickle", "wb")
    pickle.dump(file, pickle_out)
    pickle_out.close()


def creator(parnames, minmax, data):
    res = [dict(tickvals=[i for i in data['index']],
                ticktext=[str(i) for i in data['index']],
                label='index', values=data['index'])]
    for i in parnames:
        res.append(dict(range=[minmax[i].loc["min"], minmax[i].loc["max"]],
                        constraintrange=[data[i].min(), data[i].max()],
                        label=i, values=data[i]))

    res.append(dict(range=[data['like1'].min(), data['like1'].max()],
                    label='kge', values=data['like1']))
    return res


def _readFromFile(config_filename):
    """Reads a configuration from a file."""
    log = logging.getLogger(__name__)

    conf = configparser.ConfigParser()
    conf.optionxform = str
    try:
        conf.read(config_filename)
    except:
        log.error("File not found: {}".format(config_filename))
        sys.exit()
    return conf


def _parseConfig(config):
    """Parses configuration object into dictionary of options."""
    options = {}
    for section in config.sections():
        options[section] = {}
        for item in config.items(section):
            options[section][item[0]] = item[1]
    return options


def create_sampling_gridcells(nc_file):
    param = nc.Dataset(nc_file, 'r')
    param.set_auto_mask(False)
    grid_array = param.variables['gridcell'][:]
    nodata = param.variables['gridcell']._FillValue
    idx = np.vstack(np.argwhere(grid_array > nodata))
    lat = param.variables["lat"][:][idx.T[0]]
    lon = param.variables["lon"][:][idx.T[1]]
    grid_cell_ids = grid_array[tuple(idx.T)]
    param.close()
    coord = pd.DataFrame({"id": grid_cell_ids, "lat": lat, "lon": lon})
    coord.set_index("id", inplace=True)
    return coord