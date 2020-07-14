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

    df = pd.read_csv(parentDir + "/" + calName + ".csv" )

    df = df.iloc[:,2:7]
    gen =np.repeat(range(1,11),30)
    df["gen"] = gen


    #df.hist(bins=20,column="Ksat2d",by="gen")
    #plt.show()
    # n of duplicates per generation
    dup = []
    for i in range(30,(30*11),30):
        #df1  = df[df["gen"]==i]
        df1 = df.iloc[:i,:]
        dup.append(df1[df1.drop("gen",axis=1).duplicated() ].count())

    dfdup = pd.concat(dup,axis=1)
    to = 300
    ax = df.iloc[:to].drop("gen",axis=1).plot(style=".",color="black")
    plt.yscale("log")
    df.iloc[:to][df.iloc[:to].drop("gen",axis=1).duplicated()].drop("gen",axis=1).plot(style =".",ax=ax)
    plt.yscale("log")
    #df[df.drop("gen",axis=1).duplicated() ].count().plot()
    #df["depth2d"].plot.kde(bw_method=0.3)
    plt.show()
    import pdb; pdb.set_trace()
