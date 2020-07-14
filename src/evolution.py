import math
import numpy as np
import copy




class tournament_selection:

    def __init__(self,pressure = 2):
        self.pressure = pressure

    def calc(self,pop_rank):
        # number of random individuals needed

        n_select = len(pop_rank)
        n_random = n_select * self.pressure #n_select * n_parents * pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop_rank))

        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop_rank))[:n_random]
        #P = np.reshape(P, (n_select * n_parents, pressure))

        P = np.reshape(P, (n_select, self.pressure))

        n_tournament,n_competitors = P.shape

        S = np.full(n_tournament,-1,dtype=np.int)

        for i in range(n_tournament):
            a,b = P[i]

            if pop_rank[a] < pop_rank[b]:
                S[i] = a
            else:
                S[i] = b


        return S


def random_permuations(n, l):
    perms = []
    for i in range(n):
        perms.append(np.random.permutation(l))
    P = np.concatenate(perms)
    return P


class crossover:

    def __init__(self,crossProb=0.9): 

        self.crossProbThreshold = crossProb

    def calc(self,pop,n_var):

        n_pop = pop.shape[0]
        crossProbability = np.random.random((n_pop))
        do_cross = crossProbability <  self.crossProbThreshold
        R = np.random.randint(0,n_pop,(n_pop,2))
        parents = R[do_cross]
        crossPoint = np.random.randint(1,n_var,parents.shape[0])
        d = pop[parents,:]
        child = []
        for i in range(parents.shape[0]):
            child.append(np.concatenate([d[i,0,:crossPoint[i]],d[i,1,crossPoint[i]:]]))
        child = np.vstack(child)
        pop[do_cross,:] = child
        print(f"crossover applied: {np.sum(do_cross)}")
        return pop




class polynomial_mutation:

    def __init__(self,prob_mut,eta_mut):

        self.prob_mut = prob_mut
        self.eta_mut = eta_mut

    def calc(self,x,xl,xu):

        X = copy.deepcopy(x)
        Y = np.full(X.shape,np.inf)

        xl = np.asarray(xl)
        xu = np.asarray(xu)

        do_mutation = np.random.random(X.shape) < self.prob_mut

        m = np.sum(np.sum(do_mutation))
        print(f"mutants locations: {m}")

        Y[:,:] = X
        #import pdb; pdb.set_trace()

        xl = np.repeat(xl[None,:],X.shape[0],axis=0)[do_mutation] #selecting who is mutating
        xu = np.repeat(xu[None,:],X.shape[0],axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu -xl)

        mut_pow = 1.0/(self.eta_mut + 1.0)


        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)
        #import pdb; pdb.set_trace()

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta_mut + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta_mut + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]
        #import pdb; pdb.set_trace()

        _Y = X + deltaq * (xu - xl)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        Y[do_mutation] = _Y

        return Y




