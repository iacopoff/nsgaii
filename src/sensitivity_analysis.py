import os,sys

sys.path = ["/home/iff/research/dev/nsgaii/vic"] + sys.path


from cal_spotpy_functions import _parseConfig,_readFromFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import gp_emulator

def read_config(configFile):

    config_global = _readFromFile(configFile)
    vic_config = _parseConfig(config_global)['vic_config']


    return vic_config


def normalize(data):
    res = (data - data.mean()) / (data.max() - data.min())
    return res






if __name__ == "__main__":
    config = read_config("../src/config.ini")

    parentDir = config["parentDir"]
    calName = config["calOutName"]
    print(calName)

    df = pd.read_csv(parentDir + "/" + calName + ".csv" )

    # prepare tables
    params = {"depth2d":[np.random.uniform,0.1,1],
              "depth3d":[np.random.uniform,0.1,4],
              "Dsmax1d":[np.random.uniform,1,30],
              "infilt1d":[np.random.uniform,0.0001,0.4],
              "expt2d":[np.random.randint,5,30],
              "Ksat2d":[np.random.randint,100,1000]}

    bounds = [params[i][1:] for i in params]
    bounds = [[-1,1] for i in params]
    
    X = df.filter(params.keys(),axis=1)
    X = normalize(X)
    Y = df.iloc[:,1] * -1

    train = 400
    X_train = X.values[:train]
    Y_train = Y.values[:train]

    X_valid = X.values[train:]
    Y_valid = Y.values[train:]

    gp = gp_emulator.GaussianProcess (X_train,Y_train)
    gp.learn_hyperparameters(n_tries=20)


    fig = plt.figure(figsize=(12,4))

    y_pred, y_unc, _ = gp.predict(X_valid,
                              do_unc=True, do_deriv=True)


    plt.errorbar(Y_valid, y_pred,yerr=y_unc, xerr=None,elinewidth=1, fmt='.', lw=2., label="Predicted")

    plt.fill_between(y_pred, y_pred-1.96*y_unc,
                 y_pred+1.96*y_unc, color="0.8")

    import pdb;pdb.set_trace()
    plt.show()
    problem = {
    'num_vars':len(params.keys()),
    'names':list(params.keys()),
    'bounds':bounds}

    # Generate samples
    param_values = saltelli.sample(problem, 1000) 

    y,y_unc, _  = gp.predict(param_values)

    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    plt.scatter(param_values[:,0],y,color="blue",marker=".",alpha=0.01)
    ax2 =fig.add_subplot(412)
    plt.scatter(param_values[:,1],y,color="red",marker=".",alpha=0.01)
    ax3 =fig.add_subplot(413)
    plt.scatter(param_values[:,2],y,color="orange",marker=".",alpha=0.01)
    ax4 =fig.add_subplot(414)
    plt.scatter(param_values[:,3],y,color="black",marker=".",alpha=0.01)

    Si = sobol.analyze(problem, y,print_to_console=False)

    T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si)

    import pdb;pdb.set_trace()




    x = np.arange(len(params.keys()))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, first_Si["S1"], width, label='first interact',yerr=first_Si["S1_conf"])
    rects2 = ax.bar(x + width/2, T_Si["ST"], width, label='total interact',yerr=T_Si["ST_conf"])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(params.keys())
    ax.legend()


    new = {i:z for i,z in zip(Si.keys(),Si.values())}
    #np.save('SAresults.npy',Si)

    #plt.imshow(Si["S2"])

    #df = pd.DataFrame(T_Si)

    #df.index= problem["names"]
    plt.show()
