import os,sys

sys.path = ["/home/iff/research/dev/nsgaii/vic"] + sys.path

import numpy as np

from cal_spotpy_functions import _parseConfig,_readFromFile
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



def read_config(configFile):

    config_global = _readFromFile(configFile)
    vic_config = _parseConfig(config_global)['vic_config']


    return vic_config



def quickplot(result,direction="maximise",obj_function_label = None ,figsize=(20,15)):
    
    if direction == "maximise":
        Ft = -np.vstack(result.Ft)    
        F_pareto = -result.F
    elif direction == "minimise":
        Ft = np.vstack(result.Ft)  
        F_pareto = result.F

    # plot obj functions
    fig,ax = plt.subplots(Ft.shape[1],1,sharex = True,figsize=figsize)
    for i in range(result.F.shape[1]):
        ax[i].plot(Ft[:,i],linewidth=1,alpha=0.8)


    Pt = np.vstack(result.P)
    # plot parameters
    fig,ax = plt.subplots(Pt.shape[1],1,sharex=True,figsize=(20,10))
    for i in range(Pt.shape[1]):
        ax[i].plot(Pt[:,i],linewidth=1)
        ax[i].set_title(result.labels[i])
 

    fig = plt.figure()
    if Ft.shape[1] < 3:    
        ax = fig.add_subplot(111)
        x,y = Ft[:,0],Ft[:,1]
        x_pareto,y_pareto = F_pareto[:,0],F_pareto[:,1]
        ax.scatter(x_pareto,y_pareto,color="red")
        ax.scatter(x,y)
        ax.set_xlabel(obj_function_label[0])
        ax.set_ylabel(obj_function_label[1])

    elif Ft.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        x,y,z = Ft[:,0],Ft[:,1],Ft[:,2]
        x_pareto,y_pareto,z_pareto = F_pareto[:,0],F_pareto[:,1],F_pareto[:,2]
        ax.scatter(x_pareto,y_pareto,z_pareto,color="red")
        ax.scatter(x,y,z)
        ax.set_xlabel(obj_function_label[0])
        ax.set_ylabel(obj_function_label[1])
        ax.set_zlabel(obj_function_label[2])
    elif Ft.shape[1] > 3:
        ax = fig.add_subplot(111, projection='3d')
        x,y,z,v = Ft[:,0],Ft[:,1],Ft[:,2],Ft[:,3]
        x_pareto,y_pareto,z_pareto,v_pareto = F_pareto[:,0],F_pareto[:,1],F_pareto[:,2],F_pareto[:,3]
        ax.scatter(x_pareto,y_pareto,z_pareto,color="red")
        ax.scatter(x,y,z,c=v)
        ax.set_xlabel(obj_function_label[0])
        ax.set_ylabel(obj_function_label[1])
        ax.set_zlabel(obj_function_label[2])


    
    plt.show()



 


if __name__ == "__main__":
    import numpy as np
    config = read_config("../src/config.ini")

    parentDir = config["parentDir"]
    calName = config["calOutName"]
    print(calName)

    df = pd.read_csv(parentDir + "/" + calName + ".csv" )
    n_pop = 30
    df_sim = df.iloc[:,8:]

    outVar = config['vicOutVar'].split(',')

    col = [i.split(".")[0] for i in df_sim.columns]

    sims = [df_sim.filter(like="sim_"+ str(i),axis=1) for i in range(len(outVar))]
    cells = set([i.split("_")[-1].split(".")[0] for i in sims[1].columns])

    # import re

    # c0 = re.compile("sim_0")
    # c1 = re.compile("sim_1")
    
    # sim1 = df.filter([i for i in col if c0.match(i)],axis=1)

    # sim2 = df.filter([i for i in col if c1.match(i)],axis=1)
    import matplotlib.pyplot as plt

    import datetime
    dates = pd.date_range(config["calStart"],config["calEnd"],freq="M")
    for isim,sim in enumerate(sims):
        fig,ax = plt.subplots(len(cells),sharey=True)
        plt.suptitle(outVar[isim])
        for i,cell in enumerate(cells):
            ax[i].plot(dates,sim.T.filter(like=cell,axis=0))
            ax[i].set_title(cell)

        plt.tight_layout()
        plt.show()

    
    from mpl_toolkits.mplot3d import Axes3D
    df = df.iloc[-30:,:]
    df = df.reset_index().drop("index",axis=1)
    df.iloc[:,2:8] = (df.iloc[:,2:8] - df.iloc[:,2:8].mean()) / (df.iloc[:,2:8].max() - df.iloc[:,2:8].min())
    parnames = df.iloc[:,2:8].columns
    x,y = df.iloc[:,0],df.iloc[:,1]

    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111, projection='3d')

    for n,i in enumerate(range(2,8)):
        ax.scatter(x,y,df.iloc[:,i],marker="o",label=parnames[n])
        #ax.plot(x,y,df.iloc[:,i],linestyle="--")
    plt.legend()

    ax.set_xlabel(outVar[0])
    ax.set_ylabel(outVar[1])
    ax.set_zlabel("parameters")
    plt.show()

   

    import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()

