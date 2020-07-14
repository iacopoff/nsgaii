from vic import VIC
import numpy as np
from population import Pop
from dask.distributed import Client,LocalCluster
from dask.distributed import as_completed
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n,:]

def run():
    cluster = LocalCluster(n_workers=8)
    c = Client(cluster)

    parentDir = "/home/iff/research/dev/nsgaii/parallel"
    problem = VIC(parentDir=parentDir,n_var=5,n_obj=3)

    generations = 3

    init_func = np.random.random

    pop = Pop(init_func =init_func,n_pop=200,n_var=5)

    pop.initialize_population()




    n_pop = len(pop.pop)
    n_workers = len(c.nthreads())


    for igen in range(generations):

        pop_chunks = list(chunks(pop.pop,int(pop.n_pop/n_workers)))

        fut = [c.submit(problem.evaluate,p) for p in pop_chunks]


        res = as_completed(fut)

        Pt = np.vstack([i.result() for i in res])

        print(Pt)

    c.shutdown()

if __name__ == "__main__":

    run()


    
