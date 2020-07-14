import xarray as xr

import pandas as pd
import matplotlib.pyplot as plt


data = xr.open_dataset("/projects/mas1261/wp3/VIC/ba_bsn_025/run_190724_distr/fluxes_ba_bsn_.2004-01-01.nc")

d2 = data.sel(lat=13,lon=110,method="nearest")

bflow = d2.variables["OUT_BASEFLOW"]
rflow = d2.variables["OUT_RUNOFF"] +1
evap = d2.variables["OUT_EVAP"] +1
storage = d2.variables["OUT_SOIL_MOIST"]
rain = d2.variables["OUT_RAINF"] + 1
transp = d2.variables["OUT_TRANSP_VEG"] + 1
pet = d2.variables["OUT_PET"] + 1


fig = plt.figure()
plt.plot(bflow,label="bflow")
plt.plot(rflow,label="rflow")
plt.plot(evap,label="evap")
plt.plot(rain,label="rainfall")
plt.plot(pet,label="pet")
plt.plot(transp,label="transp")

for i in range(storage.shape[1]):
    plt.plot(storage[:,i],label=f"storage_{i}")
plt.legend()

ax =fig.get_axes()
ax[0].set_yscale("log")

t =d2.time.values
r = pd.DataFrame({"time":t,"baseflow":bflow.values,"runoff":rflow.values,"evap":evap.values,"storage1":storage[:,0].values,
                  "storage2": storage[:, 1].values,"storage3":storage[:,2].values,"rain":rain.values})



r["storage"] = r["storage1"] + r["storage2"] + r["storage3"]

r =r.set_index("time")


s2 = r.groupby([lambda x: x.year, lambda x: x.month]).sum()


s2 = r.groupby([lambda x: x.year]).sum()

s2["storage_delta"] = s2["storage"].diff().values


nam = list(s2.columns.values)[:-1]
nam = ["storage_delta"] + nam

name = ['storage_delta',
 'baseflow',
 'runoff',
 'evap',
 'rain']


del s2["storage"]
del s2["storage1"]
del s2["storage2"]
del s2["storage3"]

s2 = s2[name]
ax = s2.plot(kind="bar",stacked=True)
ax.set_yscale('log')

d = data.isel(lat=10,lon=10,)


d.to_dataframe()