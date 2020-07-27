# the parameter class will be read by Pop class

import scipy
from numpy import random

class Param:

    # exponential, uniform and normal distributions
    # ["name",distr,*args]
    #
    def __init__(self,param,constraint):

        if isinstance(param,dict):

            set_constraint(param,n_pop)

        if isinstance(param,list):

    def set_constraint(par,n_pop):

        c = re.compile("(?P<par1>[a-zA-z0-9]*) (?P<sign>[<=>]*) (?P<par2>[a-zA-z0-9]*)")
        ret = []
        xl_l= []
        xl_u= []
        for ipar in par:
            print(ipar)
            # loop  over each parameter and check if there is cosntraint condition
            try:

                string = par[ipar][3] # error if there is no string

                par1,rel,par2 = c.findall(string)[0]
                print(par1,rel,par2)

                func1,a1,b1 = par[par1][:3]
                func2,a2,b2 = par[par2][:3]

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

                        if len(replacement[mask]) < np.sum(fail):

                            try:
                                replacement2 = func1.rvs(a1,b1,10000)
                                mask = replacement2 > np.percentile(o2,95)
                                replacement = replacement2[mask]
                                replacement[::-1].sort()

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

            except:
                if ipar == par2:
                    ret.append(o2)
                else:
                    func1,a1,b1 = par[ipar][:3]
                    o1 = func1.rvs(a1,b1,n_pop)
                    ret.append(o1)

        pop = np.column_stack(ret)

        return pop






