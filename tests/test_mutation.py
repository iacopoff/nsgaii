import numpy as np
import copy



X = np.random.random((20,3))

ret = copy.deepcopy(X)


xl = np.array([2,4,2])
xu = np.array([10,15,10])

Y = np.full(X.shape,np.inf)

prob_mut = 0.3
eta_mut = 100
do_mutation = np.random.random(X.shape) < prob_mut


Y[:,:] = X
#import pdb; pdb.set_trace()

xl = np.repeat(xl[None,:],X.shape[0],axis=0)[do_mutation] #selecting who is mutating
xu = np.repeat(xu[None,:],X.shape[0],axis=0)[do_mutation]

X = X[do_mutation]

delta1 = (X - xl) / (xu - xl)
delta2 = (xu - X) / (xu - xl)

mut_pow = 1.0/(eta_mut + 1.0)


rand = np.random.random(X.shape)
mask = rand <= 0.5
mask_not = np.logical_not(mask)

deltaq = np.zeros(X.shape)
#import pdb; pdb.set_trace()

xy = 1.0 - delta1
val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta_mut + 1.0)))
d = np.power(val, mut_pow) - 1.0
deltaq[mask] = d[mask]

xy = 1.0 - delta2
val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta_mut + 1.0)))
d = 1.0 - (np.power(val, mut_pow))
deltaq[mask_not] = d[mask_not]
#import pdb; pdb.set_trace()

_Y = X + deltaq * (xu - xl)

_Y[_Y < xl] = xl[_Y < xl]
_Y[_Y > xu] = xu[_Y > xu]

Y[do_mutation] = _Y
import pdb;pdb.set_trace()
print(Y)
print(ret)
