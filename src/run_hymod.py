import numpy as np
from alg.nsgaii import NSGAII
from population import PopHYMOD
from problem.hymod import HYMOD,read_config
from optimize import minimize
from evolution import tournament_selection,crossover,polynomial_mutation
from parameter import Param
from visualization import quickplot
#from callbacks import PrintMutation
from dashboard import RecordEvolution




if __name__ == "__main__":

    n_pop = 30
    n_gen= 50


    params = {"cmax":{'attrs':['uniform',1,500],'bounds':[1,500],'constraint':[]},
              "bexp":{'attrs':['uniform',0.1,2.0],'bounds':[0.1,2.0],'constraint':[]},
              "alpha":{'attrs':['uniform',0.1,0.99],'bounds':[0.1,0.99],'constraint':[]},
              "Ks":{'attrs':['uniform',0.001,0.10],'bounds':[0.001,0.10],'constraint':[]},
              "Kq":{'attrs':['uniform',0.1,0.99],'bounds':[0.1,0.99],'constraint':[]}}

    config = read_config("./config_hymod.ini")

    problem = HYMOD(config)
    #problem.init_evaluation()


    pop = PopHYMOD(n_pop = n_pop,
                 params = params)


    algorithm = NSGAII(
        selection = tournament_selection(pressure=2),
        crossover = crossover(crossProb=0.9),
        mutation = polynomial_mutation(prob_mut=0.3,eta_mut = 30),
        save_history ="both",
        parallel="seq",
        cbs = [RecordEvolution()])

    result = minimize(problem=problem,
                      algorithm=algorithm,
                      population = pop,
                      n_gen=n_gen) 


    quickplot(result,direction="maximise",obj_function_label = problem.config.evalVar )
    
