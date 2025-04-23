# TSP-Evolutionary-Computing
Genetic Algorithm implementation to solve assymetric, non fully connected TS problem

Implementing a genetic algorithm tat will solve the TSP in the assymetric and not fully connected case.

--------------------------------------------------------------------------------------------------------------------------------------------------
**Representation of Candidates:** Using permutations.

**Population Initialisation:** The majority comes from Random Sampling (creating the random permutations. I have also designed a repair algorithm that convert infeasible solutions to a feasible form (not all permutations satisfy the assymetry and connectedness constraints), which is based on the two opt repair mechanism.
To have a minority that comes with a more informed way, I implemented a DFS search algorithm with backtracking, which is restricted on the k nearest neighbours in each step. It is a minority because I want a random initial population to preserve the diversity.
Also, I apply mass mutation on the best candidate to have an even better initial solution. I also estimate the best mutation operator to use (swap vs inversion) by looking on which one gave me the best final candidate.

**Selection:** Done with a quadratic ranking based method.

**Mutation:** Either swap or inversion. This is self chosen by the algorithm itself. 

**Local Search:** I apply a Local search operator after mutating. This search operator is based on k-opt search.

**Recombination Operator:** PMX.

**Elimination:** K tournament elimination. I choose a method different than the one on the selection step to avoid having too high of a selective pressure. I Also implement elitism.

**Diversity Promotion Mechanism:** I implement a method based on immigration and crowding. I put emphasis on getting rid of duplicate solutions.
--------------------------------------------------------------------------------------------------------------------------------------------------

To solve a problem, pass the corresponding csv file and set the hyperparameters in the main function of the algorithm.py
You can also check the report file where I present results, share insights and make comments about the design process and the general performance of the algorithm.
