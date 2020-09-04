import numpy as np
from alg.nsgaii import NSGAII
from population import Pop
from problem.dtlz1 import DTLZ1
from optimize import minimize
from evolution import tournament_selection,crossover,polynomial_mutation
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D





if __name__ == "__main__":

    n_obj = 3
    n_var = 5
    n_pop = 100
    n_gen= 100


    init_func= np.random.random



    problem = DTLZ1(n_var=n_var,
                    n_obj=n_obj,
                    xl=np.repeat(0,5),
                    xu=np.repeat(1,5))

    pop = Pop(n_pop = n_pop,
              n_var = n_var,
              init_func = init_func)


    algorithm = NSGAII(
        selection = tournament_selection(pressure=2),
        crossover = crossover(crossProb=0.9),
        mutation = polynomial_mutation(prob_mut=0.25,eta_mut = 30),
        save_history =True,
        parallel="blabla") 

    result = minimize(problem=problem,algorithm=algorithm,population = pop,n_gen=n_gen)

    # import pdb; pdb.set_trace()

    Ft = np.vstack(result.Ft)
    #import pdb; pdb.set_trace()
    fig,ax = plt.subplots(3,1,sharex = True,figsize=(20,10))
    for i in range(n_obj):
        ax[i].plot(Ft[:,i],linewidth=0.2,alpha=0.4)



    x,y,z = result.F[:,0],result.F[:,1],result.F[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,marker="o")
    #ax.scatter(x1,y1,z1,color="red")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.set_zlabel("f2")
    plt.show()
