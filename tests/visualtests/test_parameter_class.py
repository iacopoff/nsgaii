
from src.parameter import Param,DISTRIBUTIONS
import numpy as np
import re
import matplotlib.pyplot as plt

# data

n_pop = 500

params = {"depth2d":{'attrs':['norm',0.5,0.1],'bounds':[0.1,1],'constraint':[]},
              "depth3d":{'attrs':['norm',1.5,0.5],'bounds':[0.1,3],'constraint':["depth3d > depth2d"]},
              "Dsmax1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Wcr_FRACT2d":{'attrs':['uniform',0.3,0.55],'bounds':[0.3,0.55],'constraint':["Wcr_FRACT2d > Wpwp_FRACT2d"]},
              "Wpwp_FRACT2d":{'attrs':['uniform',0.2,0.5],'bounds':[0.2,0.5],'constraint':[]},
              "infilt1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Ksat1d":{'attrs':['randint',100,1000],'bounds':[100,1000],'constraint':["Ksat1d > Ksat2d"]},
              "Ksat2d":{'attrs':['randint',10,500],'bounds':[10,500],'constraint':[]}}


params2 = {"depth2d":{'attrs':['norm',0.5,0.5],'bounds':[0.5,0.5],'constraint':[]},
              "depth3d":{'attrs':['norm',0.5,0.5],'bounds':[0.5,0.5],'constraint':["depth3d > depth2d"]},
              "Dsmax1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Wcr_FRACT2d":{'attrs':['uniform',0.3,0.55],'bounds':[0.3,0.55],'constraint':["Wcr_FRACT2d > Wpwp_FRACT2d"]},
              "Wpwp_FRACT2d":{'attrs':['uniform',0.2,0.5],'bounds':[0.2,0.5],'constraint':[]},
              "infilt1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Ksat1d":{'attrs':['randint',10,500],'bounds':[10,500],'constraint':["Ksat1d > Ksat2d"]},
              "Ksat2d":{'attrs':['randint',10,500],'bounds':[10,500],'constraint':[]}}

# tests


def test_constraints():
    data = params

    
    par= Param()
    pop,xl,xu = par.set_constraint(data,n_pop )

    assert np.all(pop[:,1] > pop[:,0])
    assert np.all(pop[:,3] > pop[:,4])
    assert np.all(pop[:,6] > pop[:,7])

    plt.scatter(pop[:,0],pop[:,1])
    plt.plot(pop[:,0],pop[:,0])
    plt.xlabel("depth2d")
    plt.ylabel("depth3d")


    n_params = len(params.keys())
    fig,ax = plt.subplots(n_params,figsize=(20,15))
    for i,par in enumerate(params):
        func_name,a,b = params[par]["attrs"]
        func = DISTRIBUTIONS[func_name]
        
        s = func.rvs(a,b,n_pop)
        ax[i].hist(s,bins=100)
        ax[i].set_title(par)

    plt.tight_layout()
    plt.show()

test_constraints()

def test_when_params_has_equaldistribution():
    data = params2

    par= Param()
    pop,xl,xu = par.set_constraint(data,n_pop )
    n_params = len(params.keys())
    assert np.all(pop[:,1] > pop[:,0])
    assert np.all(pop[:,6] > pop[:,7])
    fig,ax = plt.subplots(n_params,figsize=(20,15))

    for i,par in enumerate(params): 
        func_name,a,b = params[par]["attrs"]
        func = DISTRIBUTIONS[func_name]
        
        s = func.rvs(a,b,n_pop)
        ax[i].hist(s,bins=100)
        ax[i].set_title(par)
    plt.show()




test_when_params_has_equaldistribution()