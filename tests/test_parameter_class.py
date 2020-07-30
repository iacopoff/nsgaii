
from src.parameter import Param,DISTRIBUTIONS
import numpy as np
import re

import pytest



# data

n_pop = 100

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



def test_outputs_dimension():

    data = params

    par= Param()
    pop,xl,xu = par.set_constraint(data,n_pop )

    n_param = len(params.keys())

    assert pop.shape == (n_pop,n_param)
    assert len(xl) == len(xu) == n_param


def test_outputs():

    data = params

    par= Param()
    pop,xl,xu = par.set_constraint(data,n_pop )

    xl_l = []
    xu_l = []
    for ipar in params:
        xl_l.append(params[ipar]['bounds'][0])
        xu_l.append(params[ipar]['bounds'][1])

    assert xl == xl_l
    assert xu == xu_l


def test_constraints():
    data = params

    
    par= Param()
    pop,xl,xu = par.set_constraint(data,n_pop )

    assert np.all(pop[:,1] > pop[:,0])
    assert np.all(pop[:,3] > pop[:,4])
    assert np.all(pop[:,6] > pop[:,7])




@pytest.mark.repeat(10)
def test_when_params_has_equaldistribution_decorator():
    data = params2

    par= Param()
    pop,xl,xu = par.set_constraint(data,n_pop )

    assert np.all(pop[:,1] > pop[:,0])
    assert np.all(pop[:,6] > pop[:,7])
