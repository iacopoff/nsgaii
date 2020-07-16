import os,sys

sys.path = ["/home/iff/research/dev/nsgaii/vic"] + sys.path


from cal_spotpy_functions import _parseConfig,_readFromFile
import pandas as pd
import matplotlib.pyplot as plt



def read_config(configFile):

    config_global = _readFromFile(configFile)
    vic_config = _parseConfig(config_global)['vic_config']


    return vic_config










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

