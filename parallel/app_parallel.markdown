# Parallelize NSGA-II

## Approaches:

The problem evaluation is the best candidate to parallelize because the population can be split into pieces and sent to each workers. The results are gathered and assembled so that the population can go through the other evolution operators.

- Master-Slave

    This seems the simplest approach, will go for it. 

- Multi-population

## Libraries:

- OpenMPI

    This seems more work to set up a Master-slave 

- Dask

    Using a LocalCluster object to set up the number of workers/processes/threads


## Spotpy:

Need to start implementing the algorithm here.

## Thoughts:

- Are there other chunks of code that can be parallelized?

- How does the configuration of the LocalCluster affects the computation? changing number of threads per worker is it affected by the GIL? Increasing the number of workers slow down each vic run execution, maybe think about a profiling?

- Optimising code: the overall nsgaii function needs profiling. The slow bits can be addressed with Cython.

## TODO:
