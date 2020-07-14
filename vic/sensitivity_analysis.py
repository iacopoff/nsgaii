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
import gp_emulator
from matplotlib import pyplot as plt
# --- sensitivity analysis


config_file = "/projects/fws1273/wp3/calibration/clusters/c1/run/test/config_multi.ini"
outdir_name = "sensitivity_analysis"
config_global = _readFromFile(config_file)
global_options = _parseConfig(config_global)

skill_threshold = None  # this is for visualisation: consider runs above/below threshold
n_best = 200       # this is just for visualisation: consider number of best runs, i will coll it behavioural
                        # later on, although it might not be the correct definition

npar = 5             # number of parameters to calibrate

sort_ascending = False  #sort the obj function value ascending from smaller values to bigger values:
                        # if you are using NSE for example, since spotpy minimize the obj function,
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

outdir_path = os.path.join(dir_work,outdir_name + "_" + cal_output_name)
# --- preprocessing


if not os.path.exists(outdir_path ):
    os.makedirs(outdir_path )



# prepare tables
cal = pd.read_csv(file_path)
parNames =  list(cal.columns[1:npar+1])
par = cal.iloc[:,0:npar+1]


x = par.drop("like1",axis=1).copy()
y = par.like1


x_train = x.values[:500]
y_train = y.values[:500]
gp = gp_emulator.GaussianProcess (x_train,y_train)
gp.learn_hyperparameters(n_tries=2)

import numpy as np
x_valid = x.values[500:600]
y_valid = y.values[500:600]

fig = plt.figure(figsize=(12,4))

y_pred, y_unc, _ = gp.predict(x_valid,
                                do_unc=True, do_deriv=False)


plt.errorbar(y_valid, y_pred,yerr=y_unc, xerr=None,elinewidth=1, fmt='.', lw=2., label="Predicted")

plt.fill_between(y_pred, y_pred-1.96*y_unc,
                    y_pred+1.96*y_unc, color="0.8")


from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

problem = {
    'num_vars':5,
    'names':['infilt1d', 'Ds1d','Dsmax1d','depth2d','depth3d'],
    'bounds':[[0.001, 0.5],
               [0.001, 0.5],
                [1,30],
               [0.1, 1.2],
               [0.5, 2.1]]}

# Generate samples
param_values = saltelli.sample(problem, 10000)

Y,y_unc, _  = gp.predict(param_values)

fig = plt.figure()
ax1 = fig.add_subplot(411)
plt.scatter(param_values[:,0],Y,color="blue",marker=".",alpha=0.01)
ax2 =fig.add_subplot(412)
plt.scatter(param_values[:,1],Y,color="red",marker=".",alpha=0.01)
ax3 =fig.add_subplot(413)
plt.scatter(param_values[:,2],Y,color="orange",marker=".",alpha=0.01)
ax4 =fig.add_subplot(414)
plt.scatter(param_values[:,3],Y,color="black",marker=".",alpha=0.01)

Si = sobol.analyze(problem, Y,print_to_console=True)

T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si)






x = np.arange(len(['infilt1d', 'Ds1d','Dsmax1d','depth2d','depth3d']))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, first_Si["S1"], width, label='first interact',yerr=first_Si["S1_conf"])
rects2 = ax.bar(x + width/2, T_Si["ST"], width, label='total interact',yerr=T_Si["ST_conf"])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(['infilt1d', 'Ds1d','Dsmax1d','depth2d','depth3d'])
ax.legend()


new = {i:z for i,z in zip(Si.keys(),Si.values())}
np.save('SAresults.npy',Si)

plt.imshow(Si["S2"])

df = pd.DataFrame(T_Si)

df.index= problem["names"]

df.plot(x=)
