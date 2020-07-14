import pymoo
from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.util.plotting import plot

#ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

#pf = get_problem("dtlz1").pareto_front(ref_dirs)
#get_visualization("scatter", angle=(45,45)).add(pf).show()

from pymoo.factory import get_sampling, get_crossover, get_mutation

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np

from pymoo.model.problem import Problem

class DTLZ1_DEV(Problem):

    def __init__(self):
        super().__init__(n_var=5,n_obj=3,xl=0,xu=1)

    def _evaluate(self,x,out,*args,**kwargs):
        
        def g1(x,k):
            return 100*( k + np.sum(np.square(x - 0.5) - np.cos(20*np.pi*(x -0.5)), axis=1))


        def dtlz1(x,n_var,n_obj):


            k = n_var - n_obj + 1

            X, X_M = x[:, :n_obj - 1], x[:, n_obj - 1:]
            g = g1(X_M,k)

            f = []
            for i in range(0,n_obj):
                _f = 0.5 * (1 + g)
                _f *= np.prod(X[:, :X.shape[1] -i],axis=1)
                if i> 0:
                    _f *= 1 - X[:,X.shape[1] -i]
                f.append(_f)
                    #import pdb; pdb.set_trace()
    
            return np.stack(f,axis=1)


        out["F"] =  dtlz1(x,n_var=5,n_obj=3)

#import pdb; pdb.set_trace()

algorithm = NSGA2(pop_size=200,
    n_offsprings=100,
    sampling=get_sampling("real_lhs"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=30),
    eliminate_duplicates=True)


problem = DTLZ1_DEV()

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               verbose=True,
               save_history=True)

print(res)

import numpy as np

history = [x.F for i in res.history for x in i.pop]

history = np.stack(history)

print(history)

plot = Scatter()
#plot.add(history,color="blue",alpha=0.05,marker=".")
#plot.add(problem.pareto_front(ref_dirs), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red",marker="o")
plot.show()

import matplotlib.pyplot as plt
import pandas as pd


fig = plt.figure()
df = pd.DataFrame(history[::10],columns= ["f1","f2","f3"])
for i,name in enumerate(df.columns):
    ax = fig.add_subplot(3,1,i +1)
    df.loc[:,name].plot(lw=0.1,figsize=(18,8),ax = ax,color="black")
    plt.title(name)
plt.show()
