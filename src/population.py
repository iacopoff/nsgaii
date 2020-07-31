import numpy as np
from parameter import Param






# population needs to have: init_population method (create the parameters given distributions and constraints)
class PopVIC(Pop):
    def __init__(self,n_pop,params,F=None,R=None):
        self.n_pop = n_pop
        self.Params = Param()
        self._F = F
        self.P = []
        self.Ft = []
        self.pop,self.xl,self.xu = self.Params.set_constraint(par=params,n_pop=n_pop)
        self.labels = list(params.keys())







class Pop:
    def __init__(self,n_pop,n_var,F=None,R=None):
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

