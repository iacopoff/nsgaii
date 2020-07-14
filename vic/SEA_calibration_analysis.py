#******************************************************************************
# FILE: calibrate_vic_sceua.py
# AUTHOR: IFF
# EMAIL:
# ORGANIZATION:
# MODIFIED BY: n/a
# CREATION DATE:
# LAST MOD DATE:
# PURPOSE:
#******************************************************************************



import sys
sys.path.append("/projects/home/iff/Dmoss/data_quality_tools/data_validation_tools")
from cal_spotpy_functions import creator
import os
import seaborn as sbn
import datetime
from itertools import chain, repeat
import matplotlib.pyplot as plt
import skill_metrics as sm
import numpy as np
from eval_functions import kge,kgenp,nse,rmse
import plotly.offline
import plotly.graph_objs as go
import pandas as pd
from cal_spotpy_functions import _parseConfig,_readFromFile, create_sampling_gridcells
import xarray as xr
import pickle
from dateutil.relativedelta import relativedelta


def extract(obs,dfgridcells,cal_start,cal_end,time_step,tslength,gridcell_index):
    print("extracting...")

    if len(gridcell_index) >=1:
        dfgridcells = dfgridcells.iloc[gridcell_index,:]
    else:
        pass

    obs["time"] = obs.indexes['time'].normalize()
    #obs = obs.loc[pd.Timestamp(cal_start):(pd.Timestamp(cal_end) + datetime.timedelta(days=1))]
    if time_step == "D":
        obs = obs.loc[pd.Timestamp(cal_start):pd.Timestamp(cal_end)]
    else:
        obs = obs.loc[pd.Timestamp(cal_start):pd.Timestamp(cal_end) ] #+ datetime.timedelta(tslength - 1)
        obs = obs.resample(time="1M").mean('time')

    dc = {}
    for n, (x, y) in enumerate(zip(dfgridcells.lon, dfgridcells.lat)):
        print(n,x,y)
        dc[n] = obs.sel(lat=y, lon=x, method="nearest").values
        if np.all(np.isnan(dc[n])) is True or np.all(np.isnan(dc[n])) is None:
            print("extracting point outside the domain: nan array returned")

    eval = pd.DataFrame(dc)
    eval.columns = dfgridcells.index
    if time_step == "D":
        eval["time"] = pd.date_range(pd.Timestamp(cal_start), pd.Timestamp(cal_end))
    else:
        eval["time"] = pd.date_range(pd.Timestamp(cal_start),pd.Timestamp(cal_end),freq="1M") #+ datetime.timedelta(tslength-1)

    return eval

# --- config

config_file = "/projects/fws1273/wp3/calibration/clusters/c2/run/smesacci/config.ini"
outdir_name = "calibration"
config_global = _readFromFile(config_file)
global_options = _parseConfig(config_global)

skill_threshold = None  # this is for visualisation: consider runs above/below threshold
n_best = 50       # this is just for visualisation: consider number of best runs, i will coll it behavioural
                        # later on, although it might not be the correct definition

npar = 5             # number of parameters to calibrate

sort_ascending = True   #sort the obj function value ascending from smaller values to larger values:
                        # if you are using NSE for example the best value is -1 (negative number because spotpy minimize the obj function)
                        # you are looking for the smaller values as the best ones

clusters = []
plot_taylor_diagrams = True



cal_output_name = global_options['vic_config']['cal_output_name']

# no edit
time_step = global_options['vic_config']['time_step']
cal_start = global_options['vic_config']['cal_start']
cal_end = global_options['vic_config']['cal_end']
dir_work = global_options['vic_config']['dir_work']
file_path = os.path.join(dir_work,cal_output_name + ".csv")
vic_out_variables = [i for i in global_options['vic_config']['vic_out_variables'].split(",")]
eval_file_names = [i for i in global_options['vic_config']['eval_file_names'].split(",")]
eval_var_names = [i for i in global_options['vic_config']['eval_var_names'].split(",")]
eval_path = global_options['vic_config']['path_eval']
outdir_path = os.path.join(dir_work,outdir_name + "_" + cal_output_name)
# --- preprocessing


if not os.path.exists(outdir_path ):
    os.makedirs(outdir_path )



# prepare tables
cal = pd.read_csv(file_path)
parNames =  list(cal.columns[1:npar+1])
par = cal.iloc[:,0:npar+1] # select parameters + perf
sim= cal.filter(regex="like1|simulation",axis=1) # select simulation results


tslength = len(sim.filter(regex="simulation1_[0-9]*",axis=1).columns)

# get parameter min and max
par_minmax = cal.loc[:,parNames].describe().loc[["min","max"],:]



obsList = []

for i,ifile in enumerate(eval_file_names):
    obsList.append(xr.open_dataset(eval_path + "/" + ifile)[eval_var_names[i]])

#obs = xr.open_dataset(eval_path + "/" +global_options['vic_config']['eval_file_names'])[global_options['vic_config']['eval_var_names']]

dfgridcells = create_sampling_gridcells(global_options['vic_config']['param_file'])

gridcell_index = open(dir_work + "/gridcell_index.pickle", "rb")
gridcell_index = pickle.load(gridcell_index)

evalList = []
for i,iobs in enumerate(obsList):
    evalList.append(extract(iobs,dfgridcells,
               cal_start=cal_start,
               cal_end=cal_end,
               time_step=time_step,
               tslength=tslength
               ,gridcell_index=gridcell_index))


# eval = extract(obs,dfgridcells,
#                cal_start=cal_start,
#                cal_end=cal_end,
#                time_step=time_step,
#                tslength=tslength
#                ,gridcell_index=gridcell_index)




max_sim = sim.drop(["like1"],axis=1).max().max()

n_cells = len(evalList[0].columns) -1

# --- plotting the selected runs based on threshold or number of best runs


# plot observed flow against calibrated flow
for d,ivar in enumerate(vic_out_variables):
    eval = evalList[d]
    max_eval = eval.drop("time", axis=1).max().max()

    river_sim = []
    river_param = []
    for s,ind in enumerate(eval.drop("time",axis=1).columns): # loop over the streams

           # extract simulation time series (no like)

        sim_best = sim.sort_values(by="like1",ascending=sort_ascending).filter(regex=f"simulation{s+1}_",axis=1)

        if n_best is not None:
            sim_best = sim_best.iloc[0:n_best, :].reset_index()
            index = sim_best["index"].values
            parbest = par.sort_values(by="like1", ascending=sort_ascending)
            df = parbest.iloc[:n_best, :]

        else:
            if sort_ascending:
                sim_best  = sim_best.loc[sim_best["like1"] <= skill_threshold,:].reset_index()
                index = sim_best["index"]
                parbest = par.sort_values(by="like1", ascending=sort_ascending)
                df = parbest.iloc[:len(index), :]
            else:
                sim_best = sim_best.loc[sim_best["like1"] >= skill_threshold, :].reset_index()
                index = sim_best["index"]
                parbest = par.sort_values(by="like1", ascending=sort_ascending)
                df = parbest.iloc[:len(index), :]


        n = len(sim_best)

        sim_best = sim_best.copy()
        s1_m = pd.melt(sim_best,id_vars="index")
        if time_step == "D":
            s1_m["variable"] = list(chain.from_iterable(zip(*repeat(pd.date_range(pd.Timestamp(cal_start),pd.Timestamp(cal_start) + datetime.timedelta(days=(tslength-1))),n ))))
        else:
            s1_m["variable"] = list(chain.from_iterable(zip(*repeat(
                pd.date_range(pd.Timestamp(cal_start), pd.Timestamp(cal_end) ,freq="1M"), #+ relativedelta(months=-1)
                n))))
        river_param.append(parbest)
        river_sim.append(s1_m)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        sbn.lineplot(data=s1_m, x="variable", y="value", ax=ax, palette=sbn.color_palette("bright", n),
                     legend=False, err_style="band",
                     ci="sd")  # ,hue="index" #,err_style="band",ci=100)#,legend="full")#, err_style="band", ci=100,)
        fig.subplots_adjust(right=0.85, left=0.05)
        fig.legend(ax.get_lines(), index, 'center right', ncol=2, bbox_to_anchor=(1, 0.8), frameon=False)
        fig.set_size_inches(29, 15)
        plt.plot(eval.time.values,eval.loc[:,ind].values, ".r-", markersize=2, linewidth=0.7)
        if time_step == "D":
            ax.set_yscale('log')
        plt.ylim([0,max(max_sim,max_eval)])
        plt.title(f"log discharge and sd bands from n {n} simulations")
        fig.savefig(
            os.path.join(outdir_path, f"timeseries_{ivar}_{cal_output_name}_{ind}.png"))
        plt.close()






# --- summary plots
for d,ivar in enumerate(vic_out_variables):
    eval = evalList[d]
    for ix,r in enumerate(eval.drop("time",axis=1).columns):
        observ =   eval.loc[:,r].values
        df2 = river_sim[ix].set_index("index")
        estim = [df2.loc[i]["value"].values for i in index]
        residual = [observ - i for  i in estim]

        ## plot summary flows
        fig = plt.figure(figsize=(29, 15))
        names = []
        ax = fig.add_subplot(211)
        for i in index:
            names.append(str(i))
            ax.plot(river_sim[ix].loc[river_sim[ix]["index"]==i,"variable"].values,river_sim[ix].loc[river_sim[ix]["index"]==i,"value"].values,label=str(i),lw=1)
            ax.plot(river_sim[ix].loc[river_sim[ix]["index"]==i,"variable"].values,observ,"r-.",lw=0.8)
        data_sorted = [np.sort(i) for  i in estim]
        data_sorted.append(np.sort(observ))
        prop = 1. * np.arange(len(data_sorted[0])) / (len(data_sorted[0]) - 1)
        ind_legend = list(index)
        ind_legend .append("obs")
        ax.legend(ncol=3, bbox_to_anchor=(1, 0.8), frameon=False)
        plt.title(ivar)


        # plot cdf log
        ax2= fig.add_subplot(224)
        for i,idata in enumerate(data_sorted):
            if i == len(ind_legend)-1:
                ax2.plot(prop, idata,'r+',label=str(ind_legend[i]))
            else:
                ax2.plot(prop, idata, label=str(ind_legend[i]))

        ax2.set_yscale('log')
        plt.title(f"cdf low flows")
        #plt.legend()


        # plot cdf high flow
        ax3= fig.add_subplot(223)
        qn = 0.95
        prop2 = prop[prop>qn]
        for i,idata in enumerate(data_sorted):
            idata = idata[prop>qn]
            if i == len(ind_legend)-1:
                ax3.plot(prop2 , idata,'r+',label=str(ind_legend[i]))
            else:
                ax3.plot(prop2 , idata, label=str(ind_legend[i]))

        plt.title(f"cdf high flows > {qn}")
        plt.ylabel("m3/d")
        #plt.legend()
        #selected_index.pop()


        fig.savefig(os.path.join(outdir_path,f"outlook_{ivar }_{cal_output_name}_{eval.columns[ix]}.png"))
        plt.close()


# # -- plot parallel coordinates
#


p = par.iloc[index].reset_index()#.filter(regex=f"like|index|[1-9]*d")

l = p.iloc[:,2:].columns.tolist()

## plot parallel plot
# this saves an image and a html file on disk. for some reason saves the files
# in the script directory.

data = [
        go.Parcoords(
            line = dict(color = p['index'],
                       colorscale = 'Jet',
                       showscale = False
                         ),
            dimensions = creator(parnames=l,minmax=par_minmax.filter(regex=f"like|index|[1-9]*d"),data=p)
        )
    ]


plotly.offline.plot({"data":data,
                         "layout": go.Layout(title="parameters")},
                         image='jpeg', filename=outdir_path +"/" + f"parallel_{cal_output_name}.html" )


