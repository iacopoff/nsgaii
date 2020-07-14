#NSGAII
import copy
import numpy as np
from numpy.random import random
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from NSGAII import dtlz1,dominates

output = "/home/iff/research/img/"
x = np.random.random((100,3))

o = dtlz1(x,n_var=5,n_obj=3)

def fastSort(x):
    n = x.shape[0]
    S = np.zeros((n,n),dtype=bool)
    Np = np.zeros(n)


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


import pandas as pd

n_var = 5
n_obj = 3
test_m = {}
test_s = {}
# for pop in range(50,500,50):
#     x = np.random.random((pop,n_var))
    
    # o = dtlz1(x,n_var=n_var,n_obj=n_obj)
    
    
    # nonDom = fastSort(o)
    # maxRank = np.max(nonDom)
    # ranks_m = {}
    # ranks_s = {}
    # for i in range(1,maxRank+1):
    #     m = np.mean(o[nonDom == i,:])
        # std = np.std(o[nonDom==i,:])
        # print(f"Rank {i}: mean {m} std {std}")
        # ranks_m["rank_" + str(i)] = m
        # ranks_s["rank_" +str(i)]  = std
    # test_m["pop_"+ str(pop)] = ranks_m
    # test_s["pop_"+ str(pop)] = ranks_s



# df_m = pd.DataFrame(test_m)
# df_s = pd.DataFrame(test_s)

# fig,ax = plt.subplots(len(df_m.columns),1,figsize=(12,15),sharex=True)
# for i,c in enumerate(df_m.columns):
#     ax[i].errorbar(x=df_m.index,y = df_m[c],yerr=df_s[c],color="black",label=c)
#     ax[i].set_title(c)
# plt.tight_layout()
#     #plt.ylabel(c)
# #df.plot()
# plt.show()


n_pop =  500

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
x = np.random.random((n_pop,n_var))

o = dtlz1(x,n_var=n_var,n_obj=n_obj)


nonDom = fastSort(o)
maxRank = np.max(nonDom) 

import matplotlib
from matplotlib import cm as cmap
norm = matplotlib.colors.Normalize(vmin=1, vmax=maxRank+2)

# # for i in range(1,maxRank+1):
#     r = o[nonDom==i,:]
#     x,y,z = r[:,0],r[:,1],r[:,2]
    #import pdb; pdb.set_trace()
#     ax.scatter(x,y,z,color=cmap.rainbow(norm(i)),label="rank" + str(i))
#     ax.set_xlabel("f0")
#     ax.set_ylabel("f1")
#     ax.set_zlabel("f2")

# plt.legend()
# plt.show()


# test crowding
from NSGAII import crowdDist

crDist = np.empty(n_pop)

fig = plt.figure(figsize=(15,15))
for rk in range(1,7):
    crDist[nonDom == rk] = crowdDist(o[nonDom==rk,:])
    r = o[nonDom==rk,:]
    ax = fig.add_subplot(3,2,rk, projection='3d')#,azim=-20)
    x,y,z = r[:,0],r[:,1],r[:,2]
    im = ax.scatter(x,y,z,c=crDist[nonDom==rk],label="rank" + str(rk),alpha=1,cmap=cmap.coolwarm)
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.set_zlabel("f2")
    ax.set_title("rank_" + str(rk))
    fig.colorbar(im, ax=ax)
plt.tight_layout(rect=[0,0,0,0.2])
plt.suptitle("Crowding distance",fontsize=20)
fig.savefig(output + "test_crowding_distance_normalized.png")
plt.show()



