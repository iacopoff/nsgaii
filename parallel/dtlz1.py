import numpy as np
import time
from pathlib import Path
from shutil import rmtree
import numpy as np

from dask.distributed import get_worker

def g1(x,k):
    return 100*(k + np.sum(np.square(x - 0.5) - np.cos(20*np.pi*(x -0.5)), axis=1))


class DTLZ1:

    def __init__(self,n_var,n_obj,xl,xu):

        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu

    def evaluate(self,x):
        print(f"WORKER ID IS: {get_worker().id}")

        indir = '/home/iff/research/dev/nsgaii/parallel/'
        workerDir = indir + get_worker().id
        Path(workerDir).mkdir(parents=True, exist_ok=True)

        time.sleep(3)
        k = self.n_var - self.n_obj + 1

        X, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = g1(X_M,k)

        f = []
        for i in range(0,self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X[:, :X.shape[1] -i],axis=1)
            if i> 0:
                _f *= 1 - X[:,X.shape[1] -i]
            f.append(_f)


        # output
        res = np.stack(f,axis=1)

        np.save(workerDir + "/result",res)

        rmtree(workerDir)

        return res
