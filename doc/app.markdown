# *Crowding distance*

Crowding distance is used to quantify the density of the solutions around each individual in the solution space.

<img src="/home/iff/research/docs/Deb_crowding_distance.png" alt="Test crowding distance"
	title="Deb's crowding distance" width="600" height="350" style="display:block;margin-left:auto;margin-right:auto"/>
    
Results from test:

<img src="/home/iff/research/img/test_crowding_distance_normalized.png" alt="Test crowding distance"
 	title="Test crowding distance" width="950" height="850" style="display:block;margin-left:auto;margin-right:auto"/>

It seems it is somehow degrading a bit the 'goodness' of some of the solutions. Look at rank 5 for example, the choice made by the crowding distance is good for ***$$f_{0}$$***
and f2 but not for f3.
# *Tournament selection*

### General genetic algorithm pseudo-code

<code python>N = population 
P = create parent population by randomly creating N individuals
while not done
    C = create empty child population
    while not enough individuals in C
        parent1 = select parent   ***** HERE IS WHERE YOU DO TOURNAMENT SELECTION *****
        parent2 = select parent   ***** HERE IS WHERE YOU DO TOURNAMENT SELECTION *****
        child1, child2 = crossover(parent1, parent2)
        mutate child1, child2
        evaluate child1, child2 for fitness
        insert child1, child2 into C
    end while
    P = combine P and C somehow to get N new individuals
end while
</code>

So then the question of how to do tournament selection can be addressed. Note that selection is only that one step of the process where we pick individuals out of the population to serve as parents of new offspring. To do so with tournament selection, you have to pick some number of possible parents, and then choose the best one as the winner. How many possible parents should be allowed to compete is the value of k I mentioned earlier.

<code>func tournament_selection(pop, k):
<br>best = null
<br>for i=1 to k
<br>&nbsp       ind = pop[random(1, N)]
<br>&nbsp       if (best == null) or fitness(ind) > fitness(best)
<br>&nbsp;&nbsp;&nbsp;&nbsp;           best = ind
<br>return best
    </code>

Let k=1. Looking at the pseudocode, this yields purely random selection. You pick one individual at random and return it.

Let k=10*N. Now we have a pretty high probability of picking every member of the population at least once, so almost every time, we're going to end up returning the best individual in the population.

Neither of these options would work very well. Instead, you want something that returns good individuals more often than bad ones, but not so heavily that it keeps picking the same few individuals over and over again. Binary tournament selection (k=2) is most often used.

In this basic framework, you can't end up with an empty population. You'll always have N individuals in the population and you'll always generate N offspring. At the end of each generation, you'll take those 2N individuals and prune them down to N again. You can either:
* throw all the parents away and just do P = C (generational replacement), 
* you can keep a few members of P and replace the rest with members of C (elitist replacement),
* you can merge them together and take the best N of the 2N total (truncation replacement),
* or whatever other scheme you come up with.


# Mutation

### Polynomial mutation

![polynomial mutation](/home/iff/research/docs/pol_mutation.png "polinomial mutation")

It is one of the many processes that simulates mutation operation on a popy 
