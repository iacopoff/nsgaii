
class Algorithm:
    """
    Base algorithm class.

    Initialise callbacks.
    """

    def __init__(self,cbs=None,**kwargs):
        self.cbs = cbs


        #init callbacks
        if self.cbs is not None: 
            for cb in self.cbs: cb.set_algorithm(self)


    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False



class GeneticAlgorithm(Algorithm):
    """
    Genetic algorithm class.

    Attributes
    ----------
    
    selection :
    crossover :
    mutation :
    save_history : str
    paralell : bool

    """
    def __init__(self,selection,crossover,mutation,save_history="both", parallel = False,**kwargs):
        """

        Parameters
        -----------
        selection :
        crossover :
        mutation :
        save_history : str, default "both"
            The evolving population history can be recorded. Three options are available: 'ram' for RAM memory, 'db' for disk memory (see Database class) or 'both' for both. 
        paralell : bool, default False

        """

        super().__init__(**kwargs)

        self.selection   = selection
        self.crossover   = crossover
        self.mutation    = mutation
        self.save_history = save_history
        self.parallel =     parallel




        if self.parallel == "dask":

            from dask.distributed import Client,as_completed,LocalCluster
            cluster = LocalCluster(n_workers=4,threads_per_worker=1,dashboard_address= ":0")
            self.client = Client(cluster)
            self.n_workers = len(self.client.nthreads())
            print(self.client.scheduler_info()['services'])
            print(self.client.dashboard_link)

    def init_pop(self,pop):
        self.pop = pop

    def init_problem(self,problem,n_gen):
        self.problem = problem
        self.n_gen = n_gen
    

    def evolve(self):
        self._evolve()



