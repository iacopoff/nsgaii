import numpy as np
from parameter import Param


class Pop:
    """
    Population class.

    Attributes
    ----------
    n_pop:
    n_var:
    _F:
    P:
    Ft:
    
    """
    def __init__(self,n_pop,n_var,F=None,R=None):
        """
        """
        self.n_pop = n_pop
        self.n_var =n_var
        self._F = F
        self.P = []
        self.Ft = []

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self,value):
        self._F = value

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self,value):
        self._R = value

    def save(self,P=None,F=None):
        if P is not None:
            self.P.append(P)

        if F is not None:
            self.Ft.append(F)

    def __len__(self):
        return len(self.pop)


    def __repr__(self):
        txt = " === \nPop shape: {} \nObj functions shape: {} \nRanks shape: {} \n ===".format(self.pop.shape,self.F.shape,self.R.shape)
        return txt



# population needs to have: init_population method (create the parameters given distributions and constraints)
class PopVIC(Pop):
    """
    Population class for VIC problem. 

    Implements parameters and sets constraints 

    Example:
    -------

        params = {"depth2d":{'attrs':['norm',0.5,0.1],'bounds':[0.1,1],'constraint':[]},
              "depth3d":{'attrs':['norm',1.5,0.5],'bounds':[0.1,3],'constraint':["depth3d > depth2d"]},
              "Dsmax1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Wcr_FRACT2d":{'attrs':['uniform',0.3,0.55],'bounds':[0.3,0.55],'constraint':["Wcr_FRACT2d > Wpwp_FRACT2d"]},
              "Wpwp_FRACT2d":{'attrs':['uniform',0.2,0.5],'bounds':[0.2,0.5],'constraint':[]},
              "infilt1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Ksat1d":{'attrs':['randint',100,1000],'bounds':[100,1000],'constraint':["Ksat1d > Ksat2d"]},
              "Ksat2d":{'attrs':['randint',10,500],'bounds':[10,500],'constraint':[]}}

        pop = PopVIC(
                    n_pop = 50,
                    params = params
                    )
    """
    def __init__(self,n_pop,params,F=None,R=None):
        self.n_pop = n_pop
        self.Params = Param()
        self._F = F
        self.P = []
        self.Ft = []
        self.pop,self.xl,self.xu = self.Params.set_constraint(par=params,n_pop=n_pop)
        self.labels = list(params.keys())