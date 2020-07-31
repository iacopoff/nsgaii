#nsga-ii algorithm
import numpy as np
from numpy.random import random,randint,uniform,permutation
from utils import fastSort,crowdDist
from db import Database,VicDriverMultiGridcell
import pandas as pd


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n,:]

class NSGAII:
    """

    """

    def __init__(self,selection,crossover,mutation,save_history=False,parallel=False):

        self.selection   = selection
        self.crossover   = crossover
        self.mutation    = mutation
        self.save_history = save_history
        self.parallel = parallel

        # TODO: instead of calling self.problem.config.whatever, create a self.config = self.problem.config

        if self.parallel == "dask":

            from dask.distributed import Client,as_completed,LocalCluster
            cluster = LocalCluster(n_workers=2,threads_per_worker=1,dashboard_address= ":0")
            self.client = Client(cluster)
            self.n_workers = len(self.client.nthreads())
            print(self.client.scheduler_info()['services'])
            print(self.client.dashboard_link)


    def init_pop(self,pop):
        self.pop = pop

    def init_problem(self,problem,n_gen):
        self.problem = problem
        self.n_gen  = n_gen


    def _evolve(self):

        for igen in range(1,self.n_gen+1):

            print(f"Generation n: {igen}")

            # initialisation
            if igen == 1:

                # initialize population
                #self.pop.init_population()

                # init evaluation
                if hasattr(self.problem,'init_evaluation'):
                    self.problem.init_evaluation()

                # evaluate population
                if self.parallel == "dask":


                    jobs =[]
                    for j in range(self.pop.n_pop):
                        jobs.append(self.client.submit(self.problem.evaluate,self.pop.pop[j],self.pop.labels))
                    res = [job.result() for job in jobs]

                    self.pop.F = np.vstack([i[0] for i in res])
                    self.sim = [i[1] for i in res]

                else:
                    jobs =[]
                    for j in range(self.pop.n_pop):
                        jobs.append(self.problem.evaluate(x=self.pop.pop[j],l=self.pop.labels))
                    self.pop.F = np.vstack(jobs) # n_var and n_obj from problem clas


                 # init db and write first population

                if self.save_history:

                    self.db = Database(
                        driver = VicDriverMultiGridcell(gridcells=self.problem.savedgridID,
                                                        param_lab =self.pop.labels),
                        obj_function=self.pop.F,
                        param=self.pop.pop,
                        simulation=self.sim,
                        connection=self.problem.config.parentDir + "/" + self.problem.config.calOutName)

                    self.db.init()

                    self.db.write()


                # non-dominance
                nonDomRank = fastSort(self.pop.F)

                # crowding distance
                crDist = np.empty(self.pop.n_pop)
                for rk in range(1,np.max(nonDomRank)+1):
                    crDist[nonDomRank == rk] = crowdDist(self.pop.F[nonDomRank==rk,:])

                # sorting
                self.pop.R =  np.lexsort((-crDist,nonDomRank)) 
                Psort = self.pop.pop[self.pop.R]

                # selection

                offsprings = self.selection.calc(pop_rank = self.pop.R)

                Qt = Psort[offsprings,:]

                # crossover

                Qt = self.crossover.calc(pop =Qt,n_var = self.problem.config.n_var)

                # mutation

                Qt = self.mutation.calc(x = Qt,xl = self.pop.xl,xu = self.pop.xu)

            else:

                # combine parent and offsprings population 

                Rt = np.vstack([self.pop.pop,Qt])

                if self.parallel == "dask":

                    # no need to evaluate again the parent population (self.pop.pop). Although also in the
                    # offsprings there are likely some duplicates, what to do with those?
                    Qt_df = pd.DataFrame(Qt)

                    Qt_dup,Qt_nodup = Qt_df[Qt_df.duplicated()].values,Qt_df[~Qt_df.duplicated()].values

                    jobs =[]
                    for j in range(Qt_nodup.shape[0]):
                        jobs.append(self.client.submit(self.problem.evaluate,Qt_nodup[j],self.pop.labels))
                    res = [job.result() for job in jobs]

                    # the vic problem return a list with 2 objects. The obj function and the simulation time series
                    # this only to be able to save the sim in the db. Better ideas here?

                    F_offspring = np.vstack([i[0] for i in res])
                    # stack the obj functions from the offspring on top of the parent obj functions
                    self.pop.F = np.vstack([self.pop.F,F_offspring])
                    self.sim = [i[1] for i in res]


                else:
                    jobs =[]
                    for j in range(Qt.shape[0]):
                        jobs.append(self.problem.evaluate(x=Qt[j],l=self.pop.labels))

                    self.pop.F = np.vstack([self.pop.F,np.vstack(jobs)]) # NO need to recalculate obj function for parent population


                # non-dominance
                nonDomRank = fastSort(self.pop.F)

                # crowding distance
                crDist = np.empty(len(self.pop.F))
                for rk in range(1,np.max(nonDomRank)+1):
                    crDist[nonDomRank == rk] = crowdDist(self.pop.F[nonDomRank==rk,:])

                # sorting
                self.pop.R =  np.lexsort((-crDist,nonDomRank))[:self.pop.n_pop]
                Psort = Rt[self.pop.R]

                self.pop.F = self.pop.F[self.pop.R]
                self.pop.pop = Psort[:,:]

                # save
                if self.save_history:
                    self.pop.save(P=Psort,F = self.pop.F)
                    self.db.write()

                    


                # selection

                offsprings = self.selection.calc(pop_rank = self.pop.R)

                Qt = Psort[offsprings,:]

                # crossover

                Qt = self.crossover.calc(pop =Qt,n_var = self.problem.config.n_var)

                # mutation

                Qt = self.mutation.calc(x = Qt,xl = self.pop.xl,xu =self.pop.xu)
