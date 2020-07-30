# the parameter class will be read by Pop class

from scipy.stats import uniform,norm,expon,gamma,beta,randint
import numpy as np
from numpy import random
import warnings
import re

class Uniform:
    def __init__(self):
        pass
    def __call__(self,a,b,size):
        return a + uniform.rvs(size=size)*(b-a)
    def rvs(self,a,b,size):
        return a + uniform.rvs(size=size)*(b-a)

class Beta:
    def __init__(self):
        pass
    def __call__(self,loc,scale,a,b,size):
        return a + beta.rvs(loc,scale,size=size)*(b-a)
    def rvs(self,loc,scale,a,b,size):
        return a + beta.rvs(loc,scale,size=size)*(b-a)

class Randint:
    def __init__(self):
        pass
    def __call__(self,a,b,size):
        return randint.rvs(low=a,high=b,size=size)
    def rvs(self,a,b,size):
        return randint.rvs(low=a,high=b,size=size)

uniform_wrap = Uniform()
beta_wrap = Beta()
randint_wrap = Randint()


DISTRIBUTIONS = {'uniform':uniform_wrap,
                 'norm':norm,
                 'expon':expon,
                 'gamma':gamma,
                 'beta':beta_wrap,
                 'randint':randint_wrap}

from copy import deepcopy

class Param:
    """
    Parameter class
    """
    def __init__(self):

        pass

    def set_constraint(self,par,n_pop):

        c = re.compile("(?P<par1>[a-zA-z0-9]*) (?P<sign>[<=>]*) (?P<par2>[a-zA-z0-9]*)")

        ret = deepcopy(par)
        xl= {i:[] for i in par.keys()}
        xu= {i:[] for i in par.keys()}



        # separate parameters that have constraints
        pairs = []    
        for ipar in par:
            if len(par[ipar]['constraint']) > 0:
                string = par[ipar]['constraint'][0]
                par1,rel,par2 = c.findall(string)[0]
                pairs.append((par1,rel,par2))

        param_with_constraints = "".join(["".join(i) for i in pairs])


        # separate parameters with no constraints and sample the distribution
        no_constr = []
        for ipar in par:
            xl[ipar] = par[ipar]['bounds'][0]
            xu[ipar] = par[ipar]['bounds'][1]

            if ipar not in param_with_constraints:
                no_constr.append(ipar)

                func1_name,a1,b1 = par[ipar]['attrs']
                func1 = DISTRIBUTIONS[func1_name]

                o1 = func1.rvs(a1,b1,n_pop)

                ret[ipar] = o1


        
        # sample distribution for parameters with constraints, apply constraints

        for ipar1,irel,ipar2 in pairs:
            
            func1_name,a1,b1 = par[ipar1]['attrs']
            func2_name,a2,b2 = par[ipar2]['attrs']

            func1 = DISTRIBUTIONS[func1_name]
            func2 = DISTRIBUTIONS[func2_name]

            o1 = func1.rvs(a1,b1,n_pop)
            o2 = func2.rvs(a2,b2,n_pop)

            if (xl[ipar1] == xl[ipar2]) and (xu[ipar1] == xu[ipar2]):
                            
                warnings.warn(f"distribution of {ipar2} and {ipar1} have same lower and upper bounds, is it correct?")
            if (a1 == a2) and (b1 == b2):
                            
                warnings.warn(f"distribution of {ipar2} and {ipar1} have same scale and loc factors, is it correct?")

            if irel == '>':
                
                count = 1
                while np.any(~(o1 > o2)):
            
                    fail = ~(o1 > o2)
                    replacement = func1.rvs(a1,b1,n_pop)
                    
                    mask =((replacement>o2) & (replacement!=o1))


                    if len(replacement[mask]) < np.sum(fail):




                        # replace o1
                        replacement2 = func1.rvs(a1,b1,10000)
                        
                        replacement = replacement2[replacement2 > np.percentile(o2,95)]

                        #replacement[::-1].sort()
                        o1[fail] = replacement[:sum(fail)]

                        # replace o2
                        mask = o2 >= np.percentile(o2,99)
                        o2[mask] = func2.rvs(a2,b2,1000)[:sum(mask)]

                        

                            


                    else:
                        
                        o1[fail] = replacement[mask][:sum(fail)]

                    count += 1

                    if count % 500 == 0: print(f"loop {count}. Failing constraints: {sum(fail)}. o1 {o1[fail]} o2 {o2[fail]}")


            ret[ipar1] = o1
            ret[ipar2] = o2


        pop = np.column_stack(list(ret.values()))

        xl = list(xl.values())
        xu = list(xu.values())

        return pop,xl,xu
