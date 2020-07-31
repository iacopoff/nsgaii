import numpy as np
from nsgaii import NSGAII
from population import PopVIC
from vic import VIC,read_config
from optimize import minimize
from evolution import tournament_selection,crossover,polynomial_mutation
from parameter import Param
from visualization import quickplot
from callbacks import PrintMutation





if __name__ == "__main__":

    n_pop = 4
    n_gen= 15



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
        parallel="dask",
        cbs = [PrintMutation()])

    result = minimize(problem=problem,
                      algorithm=algorithm,
                      population = pop,
                      n_gen=n_gen) 



    

    quickplot(result,direction="maximise",obj_function_label = problem.config.evalVar )
    