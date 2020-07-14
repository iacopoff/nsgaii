import numpy as np

class PopVIC:
    def __init__(self,n_pop,params,F=None,R=None):
        self.n_pop = n_pop
        self.params =params
        self._F = F
        self.P = []
        self.Ft = []

    def init_population(self):
        ret = []
        self.xl= []
        self.xu = []
        for par in self.params:
            func,xl,xu = self.params[par]
            self.xl.append(xl)
            self.xu.append(xu)
            ret.append(func(xl,xu,self.n_pop))

        self.pop = np.column_stack(ret)
        self.labels = list(self.params.keys())

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
        #txt = " === \nPop shape: {} \nObj functions shape: {} \nRanks shape: {} \n ===".format(self.pop.shape,self.F.shape,self.R.shape)
        #return txt
        return "test" 





class Pop:
    def __init__(self,n_pop,n_var,init_func,F=None,R=None):
        self.n_pop = n_pop
        self.n_var =n_var
        self.init_func = init_func
        self._F = F
        self.P = []
        self.Ft = []

    def initialize_population(self):
        self.pop = self.init_func((self.n_pop,self.n_var))

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

