import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #
#n_var = 3
#n_obj = 3
#x = np.random.random((1,n_var))
from pymop.problems.dtlz import DTLZ1

def g1(x,k):
    return 100*( k + np.sum(np.square(x - 0.5) - np.cos(20*np.pi*(x -0.5)), axis=1))

def eval(x,n_var,n_obj):

    k = n_var - n_obj + 1

    X, X_M = x[:, :n_obj - 1], x[:, n_obj - 1:]

    #X   =   x[:,:k]
    #X_M = x[:,k:]

    

    g = g1(X_M,k)


    f = []
    for i in range(0,n_obj):
        _f = 0.5 * (1 + g)
        _f *= np.prod(X[:, :X.shape[1] -i],axis=1)
        if i> 0:
            _f *= 1 - X[:,X.shape[1] -i]
        f.append(_f)

    return f


def eval2(x,n_var,n_obj):
    problem = DTLZ1(n_var,n_obj)
    f = problem.evaluate(x)
    return f

if __name__ == "__main__":

    #print(sys.argv)
    x = np.array(list(map(float,sys.argv[1:])))
    func = eval
    f = func(np.random.random((10000,10)),n_var=10,n_obj=5)


    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if func.__name__ == "eval":
        ax.scatter(f[0],f[1],f[2],marker=".")
    else:
        ax.scatter(f[:,0],f[:,1],f[:,2],marker=".")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.set_zlabel("f2")

    plt.show()

    
