import unittest
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np

from src.NSGAIII import crowdDist
from pymoo.algorithms.nsga2 import calc_crowding_distance

class CrowdingDistanceTest(unittest.TestCase):

    def test(x):
        return x



if __name__ == "__main__":


    cmap = cm.Greys

    #unittest.main()
    # Make data.
    x = np.exp(np.arange(0, 10, 0.5))
    y = np.exp(np.arange(0, 10, 0.5))
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    xs = X.flatten()
    ys = Y.flatten()
    zs = Z.flatten()

    d3 = np.stack([xs,ys,zs],axis=0).T

    print(d3.shape)
    d2 = np.stack([x,y],axis=0).T

    pym2 = calc_crowding_distance(d2)
    pym3 = calc_crowding_distance(d3)
    iff2 = crowdDist(d2)
    iff3 = crowdDist(d3)
    #import pdb; pdb.set_trace()


    fig = plt.figure()
    ax1 = fig.add_subplot(221,projection='3d')
    ax1.set_title("pym3")
    ax2 = fig.add_subplot(222,projection='3d')
    ax2.set_title("iff3")
    ax3 = fig.add_subplot(223)
    ax3.set_title("pym2")
    ax4 = fig.add_subplot(224)
    ax4.set_title("iff2")

    s = ax1.scatter(xs, ys, zs,c=pym3,cmap = cmap,alpha=1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    plt.colorbar(s)

    s2 = ax2.scatter(xs, ys, zs,c=iff3,cmap = cmap,alpha=1)
    ax3.scatter(x,y,c=pym2,cmap=cmap,alpha=1)

    ax4.scatter(x,y,c=iff2,cmap=cmap,alpha=1)
    plt.show()
