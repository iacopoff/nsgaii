


def minimize(problem,algorithm,population,n_gen):

    algorithm.init_problem(problem,n_gen)

    algorithm.init_pop(population)

    algorithm.evolve()

    return population
