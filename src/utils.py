import numpy as np

def dominates(a,b):
    #assert len(a.shape) > 1 and len(b.shape) > 1
    #pdb.set_trace()
    if len(a.shape) >1:
        ret = (np.sum(a <= b,axis =1) == a.shape[1]) & (np.sum(a < b,axis=1) >0)
    else:
        ret = (np.sum(a <= b) == len(a)) & (np.sum(a < b) >0)
    return ret

def crowdDist2(x):
    n = x.shape[0]

    dist = np.zeros(n)

    for obj in range(x.shape[1]):
        ord = np.argsort(x[:,obj])
        dist[ord[[0,-1]]] = np.inf
        #import pdb; pdb.set_trace()

        norm = np.max(x[:,obj]) - np.min(x[:,obj])

        for i in range(1,n-1):
            dist[i] = dist[ord[i]] + (x[ord[i+1],obj] - x[ord[i-1],obj])/norm

    return dist

def fastSort(x):
    n = x.shape[0]
    S = np.zeros((n,n),dtype=bool)
    Np = np.zeros(n)

    # DEBUG: insert some fake obj function outcome close to the minimum

    for i in range(n):
        for j in range(n):
            #import pdb; pdb.set_trace()

            S[i,j] = dominates(x[i,:],x[j,:])

    #import pdb; pdb.set_trace()
    nDom = np.sum(S,axis=0) # the n solutions that dominates i
    #import pdb; pdb.set_trace()
    Np[nDom == 0] = 1 # if i ==  0, i is non-dominated, set i rank to 1, i belongs to first non-dominated front
    k = 1
    # loop over pareto fronts
    while np.sum(Np == 0) > 0:
        #import pdb; pdb.set_trace()
        l = np.arange(n)[Np==k] # first non-dominated front
        for i in l: # loop over the non-dominated front
            nDom[S[i,:]] = nDom[S[i,:]] -1 # reduce by 1 the rank of the solutions that i dominates
        #import pdb; pdb.set_trace()
        k += 1
        # now nDom has been reduced by 1, so the next non-dominated front will be nDom ==  0
        # and Np == 0 ensure that we don't pass over the first ranked non-dom solutions
        Np[(nDom == 0) & (Np == 0) ] = k
    #import pdb; pdb.set_trace()
    
    return Np.astype(int)


def crowdDist(x):
    n = x.shape[0]

    nobj = x.shape[1]

    dist = np.zeros(n)


    ord = np.argsort(x,axis=0)
        #import pdb; pdb.set_trace()

    X = x[ord,range(nobj)]

        #import pdb; pdb.set_trace()
    
    dist = np.vstack([X,np.full(nobj,np.inf)]) - np.vstack([np.full(nobj,-np.inf),X])

    
    norm = np.max(X,axis=0) - np.min(X,axis=0)
    dist_to_last,dist_to_next = dist, np.copy(dist)
    dist_to_last,dist_to_next = dist_to_last[:-1]/norm ,dist_to_next[1:]/norm
    J = np.argsort(ord,axis=0)
    _cd = np.sum(dist_to_last[J, np.arange(nobj)] + dist_to_next[J, np.arange(nobj)], axis=1) / nobj


    return _cd
