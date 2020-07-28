import numpy as np
from nsgaii import NSGAII
from population import PopVIC
from vic import VIC,read_config
from optimize import minimize
from evolution import tournament_selection,crossover,polynomial_mutation
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from parameter import Param





if __name__ == "__main__":

    n_pop = 4
    n_gen= 15

    # TODO: create a parameter parent class. It should have methods and attributes that set the general interface
    # for child classes. For example it should have parameter name, distribution (default is uniform) and arguments
    # passed to the distribution (min,max or average and std for normal distribution). Use distribution from scipy.stats.
    # It should also define the relationships between parameters (constrained)

    params = {"depth2d":{'attrs':['norm',0.5,0.1],'bounds':[0.1,1],'constraint':[]},
              "depth3d":{'attrs':['norm',1.5,0.5],'bounds':[0.1,3],'constraint':["depth3d > depth2d"]},
              "Dsmax1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Wcr_FRACT2d":{'attrs':['uniform',0.3,0.55],'bounds':[0.3,0.55],'constraint':["Wcr_FRACT2d > Wpwp_FRACT2d"]},
              "Wpwp_FRACT2d":{'attrs':['uniform',0.2,0.5],'bounds':[0.2,0.5],'constraint':[]},
              "infilt1d":{'attrs':['uniform',1,30],'bounds':[1,30],'constraint':[]},
              "Ksat1d":{'attrs':['randint',100,1000],'bounds':[100,1000],'constraint':["Ksat1d > Ksat2d"]},
              "Ksat2d":{'attrs':['randint',10,500],'bounds':[10,500],'constraint':[]}}

    config = read_config("./config.ini")

    problem = VIC(config)


    pop = PopVIC(n_pop = n_pop,
                 params = params)

    # TODO: add option for paralel
    algorithm = NSGAII(
        selection = tournament_selection(pressure=2),
        crossover = crossover(crossProb=0.9),
        mutation = polynomial_mutation(prob_mut=0.3,eta_mut = 30),
        save_history =True,
        parallel="dask") 

    result = minimize(problem=problem,algorithm=algorithm,population = pop,n_gen=n_gen)

    #import pdb; pdb.set_trace()

    Ft = -np.vstack(result.Ft)
    #import pdb; pdb.set_trace()
    fig,ax = plt.subplots(2,1,sharex = True,figsize=(20,10))
    for i in range(result.F.shape[1]):
        ax[i].plot(Ft[:,i],linewidth=1,alpha=0.8)



    Pt = np.vstack(result.P)

    fig,ax = plt.subplots(Pt.shape[1],1,sharex=True,figsize=(20,10))
    for i in range(Pt.shape[1]):
        ax[i].plot(Pt[:,i],linewidth=1)
        ax[i].set_title(result.labels[i])
 


    x,y = -result.F[:,0],-result.F[:,1]#,result.F[:,2]
    x1,y1 = Ft[:,0],Ft[:,1]
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    ax.scatter(x1,y1,color="red")
    ax.scatter(x,y,marker="o")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    #ax.set_xlim([0,1])
    #ax.set_ylim([0,1])
    #ax.set_zlabel("f2")
    plt.show()
    import pdb; pdb.set_trace()
 
