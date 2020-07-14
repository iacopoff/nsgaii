import numpy as np
import math
from numpy.random import random

from NSGAII import dtlz1,fastSort,dominates,g1,tournament_selection_v2


pressure = 2 # default to binary tournement
n_select = 30 
n_parents = 2
elitism = n_select *n_parents

n_pop = 200
n_var = 5
n_obj = 3

class Pop:
    def __init__(self,n_pop,n_var,F=None,R=None):
        self.pop = np.random.random((n_pop,n_var))
        self._F = F

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


    def __len__(self):
        return len(self.pop)


    def __repr__(self):
        txt = " === \nPop shape: {} \nObj functions shape: {} \nRanks shape: {} \n ===".format(self.pop.shape,self.F.shape,self.R.shape)
        return txt

pop = Pop(n_pop,n_var)

Ot = dtlz1(pop.pop,n_var=n_var,n_obj=n_obj)
pop.F = Ot
pop.R = fastSort(pop.F)



def random_permuations(n, l):
    perms = []
    for i in range(n):
        perms.append(np.random.permutation(l))
    P = np.concatenate(perms)
    return P

def tournament_selection_v0(pop_rank,elitism):
    # number of random individuals needed

    eli2 = (len(pop_rank) - elitism)
    n_random = eli2 * pressure#n_select * n_parents * pressure

    # number of permutations needed
    n_perms = math.ceil(n_random / len(pop_rank))

    # get random permutations and reshape them
    P = random_permuations(n_perms, len(pop_rank))[:n_random]
    #P = np.reshape(P, (n_select * n_parents, pressure))

    P = np.reshape(P, (eli2, pressure))

    n_tournament,n_competitors = P.shape

    S = np.full(n_tournament,-1,dtype=np.int)

    for i in range(n_tournament):
        a,b = P[i]

        if pop_rank.R[a] < pop_rank.R[b]:
            S[i] = a
        else:
            S[i] = b
    return S

S = tournament_selection_v0(pop_rank=pop,elitism=elitism)

print("mean rank all population:\n")
print(f"mean: {np.mean(pop.F)}\n")

print("function 1\n")
print(f"mean:{np.mean(pop.F[S])}")
print(f"number of offsprings: {len(S)}")
print(f"duplicated offsprings: {len(S) - len(np.unique(S))}\n")

print("function 2\n")
child = tournament_selection_v2(n_pop,elitism =n_select*n_parents,pressure=2 )
print(f"mean: {np.mean(pop.F[child])}")
print(f"number of offsprings: {len(child)}")
print(f"duplicated offsprings: {len(child) - len(np.unique(child))}")
