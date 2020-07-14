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
from cal_spotpy_functions import _parseConfig,_readFromFile


# --- config

config_file = "/projects/fws1273/wp3/calibration/clusters/c1/run/test/config.ini"
outdir_name = "calibration"
config_global = _readFromFile(config_file)
global_options = _parseConfig(config_global)

skill_threshold = None  # this is for visualisation: consider runs above/below threshold
n_best = 5        # this is just for visualisation: consider number of best runs, i will coll it behavioural
                        # later on, although it might not be the correct definition

npar = 6              # number of parameters to calibrate

sort_ascending = True   #sort the obj function value ascending from smaller values to bigger values:
                        # if you are using NSE for example, since spotpy minimize the obj function,
                        # you are looking for the smaller values as the best ones

clusters = []
plot_taylor_diagrams = True


cal_output_name = global_options['vic_config']['cal_output_name']

# no edit
discharge_summary = global_options['vic_config']['discharge_summary']
discharge_folder = global_options['vic_config']['discharge_folder']
river_reachid = [int(i) for i in global_options['vic_config']['river_reachid'].split(",")]
cal_start = global_options['vic_config']['cal_start']
cal_end = global_options['vic_config']['cal_end']
dir_work = global_options['vic_config']['dir_work']
file_path = os.path.join(dir_work,cal_output_name + ".csv")

outdir_path = os.path.join(dir_work,outdir_name)
# --- preprocessing


if not os.path.exists(outdir_path ):
    os.makedirs(outdir_path )



# prepare tables
cal = pd.read_csv(file_path)
parNames =  list(cal.columns[1:npar+1])
par = cal.iloc[:,0:npar+1] # select parameters + perf
sim= cal.filter(regex="like1|simulation",axis=1) # select simulation results

# get parameter min and max
par_minmax = cal.loc[:,parNames].describe().loc[["min","max"],:]

# read observation
print("get obs")
df = pd.read_excel(discharge_summary)
selection = df.loc[df['routing_id'].isin(river_reachid)]
print(" cal analysis with " + str(selection['station'].tolist()))

# save routing id and river name
routing_info = selection[['station','routing_id']].set_index("routing_id").to_dict()



obsSeries = pd.DataFrame()
for i in river_reachid:
    id = selection['routing_id'] == i
    valFile = selection[id]['excel_name'].tolist()[0]
    obs_in_t = pd.read_csv(discharge_folder + "/" + valFile + ".csv", usecols=[0, 1])
    obs_in_t.columns = ["time", i]
    try:
        obs_in_t['time'] = pd.to_datetime(obs_in_t.time, format="%d/%m/%Y")
    except:
        obs_in_t['time'] = pd.to_datetime(obs_in_t.time, format="%Y-%m-%d")
    obs_in_t = obs_in_t[(obs_in_t.time >= pd.Timestamp(cal_start)) & (obs_in_t.time <= pd.Timestamp(cal_end))]
    obs_in_t = obs_in_t.set_index("time")
    obsSeries[i] = obs_in_t[i]




# --- plotting the selected runs based on threshold or number of best runs


# plot observed flow against calibrated flow

river_sim = []
river_param = []
for s in range(len(river_reachid)): # loop over the streams
    print(f"{river_reachid[s]}")
       # extract simulation time series (no like)
    if len(river_reachid)>1:
        sim_best = sim.sort_values(by="like1",ascending=sort_ascending).filter(regex=f"simulation{s+1}",axis=1)

    else:
        sim_best = sim.sort_values(by="like1", ascending=sort_ascending).filter(regex=f"simulation",axis=1)

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

    s1_m["variable"] = list(chain.from_iterable(zip(*repeat(pd.date_range(pd.Timestamp(cal_start),pd.Timestamp(cal_end )),n ))))
    river_param.append(parbest)
    river_sim.append(s1_m)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    sbn.lineplot(data=s1_m, x="variable", y="value", ax=ax, palette=sbn.color_palette("bright", n),
                 legend=False, err_style="band",
                 ci=99)  # ,hue="index" #,err_style="band",ci=100)#,legend="full")#, err_style="band", ci=100,)
    fig.subplots_adjust(right=0.85, left=0.05)
    fig.legend(ax.get_lines(), index, 'center right', ncol=2, bbox_to_anchor=(1, 0.8), frameon=False)
    fig.set_size_inches(29, 15)
    plt.plot(obsSeries[river_reachid[s]], ".r-", markersize=2, linewidth=0.7)
    ax.set_yscale('log')
    plt.title(f"log discharge and CI from n {n} simulations")
    fig.savefig(
        os.path.join(outdir_path, f"discharge_{cal_output_name}_{routing_info['station'][river_reachid[s]]}.png"))
    plt.close()






# --- summary plots

for r in range(len(river_reachid)):
    observ =   obsSeries[river_reachid[r]].values
    df2 = river_sim[r].set_index("index")
    estim = [df2.loc[i]["value"].values for i in index]
    residual = [observ - i for  i in estim]

    ## plot summary flows
    fig = plt.figure(figsize=(29, 15))
    names = []
    ax3 = fig.add_subplot(211)
    for i in index:
        names.append(str(i))
        ax3.plot(river_sim[r].loc[river_sim[r]["index"]==i,"value"].values,label=str(i),lw=1)
        ax3.plot(observ,"r-.",lw=0.8)

    data_sorted = [np.sort(i) for  i in estim]
    data_sorted.append(np.sort(observ))
    prop = 1. * np.arange(len(data_sorted[0])) / (len(data_sorted[0]) - 1)
    ind_legend = list(index)
    ind_legend .append("obs")

    # plot cdf log
    ax2= fig.add_subplot(224)
    for i,idata in enumerate(data_sorted):
        if i == len(ind_legend)-1:
            ax2.plot(prop, idata,'r+',label=str(ind_legend[i]))
        else:
            ax2.plot(prop, idata, label=str(ind_legend[i]))

    ax2.set_yscale('log')
    plt.title(f"cdf low flows")
    plt.legend()


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
    plt.legend()
    #selected_index.pop()


    fig.savefig(os.path.join(outdir_path,f"outlook_{cal_output_name}_{routing_info['station'][river_reachid[r]]}.png"))
    plt.close()




# -- plot parallel coordinates

for c in clusters:

    p = par.iloc[index].reset_index().filter(regex=f"like|index|[1-9]*d.{c}")

    l = p.filter(regex=f"[1-9]*d.{c}").columns.tolist()

## plot parallel plot
# this saves an image and a html file on disk. for some reason saves the files
# in the script directory.

    data = [
        go.Parcoords(
            line = dict(color = p['index'],
                       colorscale = 'Jet',
                       showscale = False
                         ),
            dimensions = creator(parnames=l,minmax=par_minmax.filter(regex=f"like|index|[1-9]*d.{c}"),data=p)
        )
    ]


    plotly.offline.plot({"data":data,
                         "layout": go.Layout(title="parameters")},
                         image='jpeg', filename=outdir_path +"/" + f"parallel_{cal_output_name}_cluster{c}.html" )



# talyor diagrams
# taylor diagram is perfect to decompose the skill metrics

if plot_taylor_diagrams:

    for ir,r in enumerate(river_sim):

        ests = []
        obss = []
        names = []
        ind = [i for i in index]
        for i in index: #s1_m["index"].unique():
            ests.append(r.loc[r["index"]==i,"value"].values)
            obss.append(obsSeries.values[:,ir])
            names.append(str(i))
            #plt.plot(s1_m.loc[s1_m["index"]==i,"value"].values,label=str(i))
            #plt.plot(obsSeries.values,"r-.")
        #plt.legend()

        # calculate statistics
        taylor_stats = [sm.taylor_statistics(ests[i], obss[i], 'data') for i in range(len(obss))]
        sdev = np.array([i["sdev"][0] if i == 0 else i["sdev"][1] for i in taylor_stats])
        crmsd = np.array([i["crmsd"][0] if i == 0 else i["crmsd"][1] for i in taylor_stats])
        ccoef = np.array([i["ccoef"][0] if i == 0 else i["ccoef"][1] for i in taylor_stats])

        # these are some skill metrics to map colours on the taylor diagram
        kge1 = np.asarray([float(kge(ests[i], obss[i])[0]) for i in range(len(obss))])
        kge2 = np.asarray([float(kgenp(ests[i], obss[i])[0]) for i in range(len(obss))])
        nse_log1 = np.asarray([float(nse(np.log(ests[i]), np.log(obss[i]))) for i in range(len(obss))])
        nse1 = np.asarray([float(nse(ests[i], obss[i])) for i in range(len(obss))])
        rmse1 = np.asarray([float(rmse(ests[i], obss[i])) for i in range(len(obss))])


        # plot different skill metrics
        label = ['Non-Dimensional Observation']
        label.extend(names)
        l = ["kge","kge_nonpar","nse","rmse","nse_log"]
        for i,fc in enumerate([kge1,kge2,nse1,rmse1,nse_log1]):
            plt.figure(figsize=(14,14))
            sm.taylor_diagram(sdev, crmsd, ccoef, styleOBS='-',
                          colOBS='r', markerobs='o',
                          titleOBS='observation', markerLabel=label,markerColor='r',
                          locationColorBar='EastOutside',
                          markerDisplayed='colorBar', titleColorBar=l[i],
                          markerLabelColor='black', markerSize=5, cmapzdata=fc,
                          )

            # Write plot to file
            plt.savefig(os.path.join(outdir_path,f"taylor_{l[i]}_{cal_output_name}_{routing_info['station'][river_reachid[ir]]}.png"))
            plt.close()
        # plot labels (indeces to the simulation runs)
        plt.figure()
        sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label, markerColor='r',
                          styleOBS='-', colOBS='r', markerobs='o',
                          markerSize=2, showlabelsRMS='on',
                          titleRMS='on', titleOBS='Ref', checkstats='on')
        plt.savefig(os.path.join(outdir_path,f"Tlabels_{cal_output_name}_{routing_info['station'][river_reachid[ir]]}.png"))
        plt.close()
else:
    pass

# ---- save parameters to file

import pickle
#
#
o = df.drop(["like1"],axis=1)
o.columns= [ i[3:] for i in parNames]
dout = o.iloc[0,:].to_dict()
#
pickle_out = open(f"{outdir_path}/bestcalparset_{cal_output_name}.pickle","wb")
#
pickle.dump(dout, pickle_out)
pickle_out.close()
