# the parameter class will be read by Pop class

from scipy.stats import uniform,norm,expon,gamma,beta,randint
import numpy as np
from numpy import random
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


class Param:

    def __init__(self):

        pass

    def set_constraint(self,par,n_pop):

        c = re.compile("(?P<par1>[a-zA-z0-9]*) (?P<sign>[<=>]*) (?P<par2>[a-zA-z0-9]*)")
        ret = []
        xl_l= []
        xu_l= []
        par1 = None
        par2 = None

        for i,ipar in enumerate(par):
            # loop over each parameter and check if there is cosntraint condition
            xl_l.append(par[ipar]['bounds'][0])
            xu_l.append(par[ipar]['bounds'][1])

            if len(par[ipar]['constraint']) > 0: # constraint

                string = par[ipar]['constraint'][0] 

                par1,rel,par2 = c.findall(string)[0]
                print(par1,rel,par2)

                func1_name,a1,b1 = par[par1]['attrs']
                func2_name,a2,b2 = par[par2]['attrs']

                func1 = DISTRIBUTIONS[func1_name]
                func2 = DISTRIBUTIONS[func2_name]

                o1 = func1.rvs(a1,b1,n_pop)
                o2 = func2.rvs(a2,b2,n_pop)

                #sbn.kdeplot(o1,kernel="tri",color="red")
                #sbn.kdeplot(o2,kernel="tri")
                #plt.figure()
                #plt.scatter(o1,o2)

                if rel == '>':
                    print(f"apply constraint {string}")
                    count = 1
                    while np.any(~(o1 > o2)):
                        fail = ~(o1 > o2)
                        replacement = func1.rvs(a1,b1,n_pop)
                        mask =((replacement>o2) & (replacement!=o1)) #[:len(fail)]
                        #import pdb; pdb.set_trace()

                        if len(replacement[mask]) < np.sum(fail):

                            try:
                                replacement2 = func1.rvs(a1,b1,10000)
                                mask = replacement2 > np.percentile(o2,95)
                                replacement = replacement2[mask]
                                #replacement[::-1].sort()

                                o1[fail] = replacement[:sum(fail)]
                            except Exception as e:
                                print("Are the parameters correct?")
                                print(e)
                                break
                        else:

                            try:
                                o1[fail] = replacement[mask][:sum(fail)]
                            except Exception as e:
                                print("Are the parameters correct?")
                                print(e)
                        count += 1
                        if count % 500 == 0: print(f"loop {count} and fail {sum(fail)}")

                ret.append(o1)

            else: # no contraint
                    
                if ipar == par2 and par2 is not None:
                    ret.append(o2)
                else:
                    func1_name,a1,b1 = par[ipar]['attrs']
                    func1 = DISTRIBUTIONS[func1_name]
                    o1 = func1.rvs(a1,b1,n_pop)
                    ret.append(o1)

        pop = np.column_stack(ret)

   
        return pop,xl_l,xu_l





