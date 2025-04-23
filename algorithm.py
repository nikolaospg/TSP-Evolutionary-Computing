import Reporter
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import numba
from numba import njit


## BASIC FUNCTIONS ##
## INITIALISATION FUNCTIONS ##
#Function thats gives us a random candidate (random permutation). It is not necessarily feasible
def get_random_candidate(current_data):
    num_cities=len(current_data)
    random_candidate=np.random.permutation(num_cities)
    random_candidate=np.append(random_candidate, random_candidate[0])	
    return random_candidate

#Uses kopt repair continuously until it returns a candidate
def get_kopt_cand(current_data):
    while(1):
        current_cand=get_random_candidate(current_data)
        if(is_feasible(current_cand, current_data)):
            return current_cand
        repaired=two_opt_repair(current_cand, current_data)
        if(repaired is not None and is_feasible(repaired, current_data)):
            return repaired


#This function gets random candidates. If they are infeasible, it applies kopt repair to fix them. If they cannot be fixed they are discarded
def kopt_initialisation(current_data, pop_size):
    population=[]
    while(len(population)<pop_size):
        current_candidate=get_random_candidate(current_data)
        if(is_feasible(current_candidate, current_data)):
            population.append(current_candidate)
        else:
            fixed_candidate=two_opt_repair(current_candidate, current_data, 10000)
            if(fixed_candidate is None):
                continue 
            else:
                population.append(fixed_candidate)
    return np.array(population)


def knn_initialisation(current_data, pop_size, max_attempts_per_candidate=1000):
    #this function uses the get_knn_candidate_with_backtracking.
    #it gives us 250 candidates using this logic, 50 candidate from each k (1,2,3,4,5).
    #it then applies mass mutation to the best one of them for 10 secs, using each one of the 2 mutation types
    #this way we estimate the best mutation type for this dataset

    #First getting the 50 knn elements
    population = []
    objs =[]
    for k in [1,2,3,4,5]:
        current_num_cands=0
        while(current_num_cands<50):
            current_candidate = get_knn_candidate_with_backtracking(current_data, k, max_attempts_per_candidate)
            if current_candidate is not None and is_feasible(current_candidate, current_data):
                population.append(current_candidate)
                objs.append(obj_fun(current_candidate, current_data))
                current_num_cands +=1 




    # finding the best candidate:
    population = np.array(population)
    objs = np.array(objs)
    best_index = np.argmin(objs)
    best_candidate = population[best_index]

    # For inversion
    inv_best= best_candidate.copy()
    t1=time.time()
    while(time.time() - t1 < 10):
        
        current_mut = mutation_operator(inv_best, "inversion")
        
        if(not is_feasible(current_mut, current_data)):
            rep=two_opt_repair(current_mut, current_data)
            if(rep is not None and is_feasible(rep, current_data)):     #This means it is fixed
                current_mut=rep
            else:
                continue
        if(obj_fun(current_mut, current_data) < obj_fun(inv_best, current_data)):
            inv_best = current_mut
    

    #For swap
    swap_best= best_candidate.copy()
    t1=time.time()
    while(time.time() - t1 < 10):
        
        current_mut = mutation_operator(swap_best, "swap")
        time_passed=time.time() - t1

        if(not is_feasible(current_mut, current_data)):
            rep=two_opt_repair(current_mut, current_data)
            if(rep is not None and is_feasible(rep, current_data)):     #This means it is fixed
                current_mut=rep
            else:
                continue
        if(obj_fun(current_mut, current_data) < obj_fun(swap_best, current_data)):
            swap_best = current_mut


    #Getting the mutation type 
    if(obj_fun(swap_best, current_data) < obj_fun(inv_best, current_data)):
        total_best=swap_best
        chosen_mut_type = "swap"
    else:
        total_best=inv_best
        chosen_mut_type = "inversion"


    #For "hard" datasets : i.e. datasets with >500 cities and missing edges, we add the extra 2 mutated candidtes
    num_cities=len(current_data)
    has_missing=np.isinf(current_data).any()
    if(num_cities>500 and has_missing == True):
        population = np.vstack([population, swap_best, inv_best])

    return population, chosen_mut_type


def get_knn_candidate_with_backtracking(current_data, k, max_attempts):

    # I use the concept of stack for backtracking. The state has 4 elements, (current_tour, unvisited, current_city, next_options)
    # current_tour -> The candidate up until the point the state was created
    # unvisited ->The unvisited cities up until that point 
    # current_city -> the final city of the tour 
    # next_options the k nearest neighbours to visit 

    def get_k_nearest(current_data, current_city, unvisited, k):

        #Helper function, it uses elements from the stack state to help us get the nearest unvisited neighbours of a city. these will be the next options
        unvisited_cities = list(unvisited)
        if not unvisited_cities:
            return []
        distances = current_data[current_city][unvisited_cities]
        sorted_indices = np.argsort(distances)
        k_nearest = [unvisited_cities[i] for i in sorted_indices[:k]] if len(unvisited_cities) >= k else unvisited_cities
        return k_nearest


    num_cities = current_data.shape[0]
    start_city = np.random.randint(num_cities)

    
    #Initialise the stack, craft the first state and push it in
    stack = deque()
    initial_unvisited = set(range(num_cities))
    initial_unvisited.remove(start_city)
    initial_tour = [start_city]
    initial_current_city = start_city
    initial_next_options = get_k_nearest(current_data, start_city, initial_unvisited, k)
    stack.append((initial_tour, initial_unvisited, initial_current_city, initial_next_options))

    attempts = 0

    #While the stack is not empty and the max attempt number has not been reached:
    while stack and attempts < max_attempts:
        current_tour, unvisited, current_city, next_options = stack.pop()

        if not next_options and not unvisited:
            # We visited all the cities, we will get the final tour 
            complete_tour = current_tour + [start_city]
            if is_feasible(complete_tour, current_data):            #Again feasibility check 2 be sure
                return np.array(complete_tour)
            else:                               #If not feasible we must backtrack
                attempts += 1
                continue  

        if not next_options:
            # no other options backtrack
            attempts += 1
            continue

        # Iterate through possible next cities
        for next_city in next_options:
            if next_city in unvisited:      #pickin a city and crafting the new state
                new_tour = current_tour + [next_city]
                new_unvisited = unvisited.copy()
                new_unvisited.remove(next_city)
                new_current_city = next_city
                new_next_options = get_k_nearest(current_data, new_current_city, new_unvisited, k)

                # Save the current state with remaining options (i.e. the other neighbours) for backtracking
                remaining_options = next_options.copy()
                remaining_options.remove(next_city)
                if remaining_options:
                    stack.append((current_tour, unvisited, current_city, remaining_options))

                # Push the new state to the stack
                stack.append((new_tour, new_unvisited, new_current_city, new_next_options))
                break  

        attempts += 1

    return None  # No feasible tour found within the maximum number of attempts




## NUMBA FUNCTIONS ##
@njit
def pmx_crossover(parent1, parent2):
    #Numba function to implement pmx operation.
    size = len(parent1)

    # Choosing two random crossover points
    point1 = np.random.randint(0, size)
    point2 = np.random.randint(0, size)
    if point1 > point2:
        point1, point2 = point2, point1


    # initialising offspring
    offspring1 = -np.ones(size, dtype=np.int64)
    offspring2 = -np.ones(size, dtype=np.int64)
        
    # Copy the segment between crossover points
    offspring1[point1:point2] = parent1[point1:point2]
    offspring2[point1:point2] = parent2[point1:point2]
        
    #function to find the position of a gene in an array
    def find_position(wanted_value, input_array):
        for index in range(len(input_array)):
            if input_array[index] == wanted_value:
                return index
        return -1  # Return -1 if the value is not found
    

    # function to Map the values from the segment in the opposite parent
    def pmx_fill(offspring, segment, other_parent):
        for i in range(point1, point2):
            if other_parent[i] not in segment:
                current_pos = i
                while offspring[current_pos] != -1:
                    current_pos= find_position(offspring[current_pos], other_parent)
                offspring[current_pos] = other_parent[i]
        
    # calling the function above
    pmx_fill(offspring1, parent1[point1:point2], parent2)
    pmx_fill(offspring2, parent2[point1:point2], parent1)
        
    # Fill in remaining positions with genes from the opposite parent
    offspring1[offspring1 == -1] = parent2[offspring1 == -1]
    offspring2[offspring2 == -1] = parent1[offspring2 == -1]

    return np.append(offspring1, offspring1[0]), np.append(offspring2, offspring2[0])

@njit
def obj_fun(current_candidate, current_data):   
    #Numba "optimised" function to calculate obj fun
    total_distance = 0.0
    for i in range(len(current_candidate) - 1):
        total_distance += current_data[current_candidate[i], current_candidate[i + 1]]
    return total_distance

@njit
def is_feasible(current_candidate, current_data):
    #Numba "optimised" function to check feasibility
    if(current_candidate is None):
        return False
    for i in range(len(current_candidate) - 1):
        if current_data[current_candidate[i], current_candidate[i + 1]] == np.inf:
            return False
    return True

@njit
def two_opt_repair(input_tour, current_data, max_iterations=1000):

    #First 2 numba friendly functions to help the two_opt_repair:
    def two_opt_swap(input_cand, i, k):
        #Numba friendly two opt swap. The segment from i up till (and including) k is reversed, the rest are copied
        cand_size = len(input_cand)
        new_cand = np.empty(cand_size, dtype=np.int64)

        # First part is basically copied
        for index in range(i):
            new_cand[index] = input_cand[index]

        # this part is reversed
        for index in range(i, k + 1):
            new_cand[index] = input_cand[k - (index - i)]

        # this again copied
        for index in range(k + 1, cand_size ):
            new_cand[index] = input_cand[index]

        # Ensure the last city equals the first
        new_cand[-1] = new_cand[0]

        return new_cand

    def get_infeasible_edges(my_tour, current_data):
        infeas_indices=[]
        for index in range(len(my_tour) -1):
            if(current_data[my_tour[index], my_tour[index+1]] == np.inf):
                infeas_indices.append(index)
        return np.array(infeas_indices)

    current_tour=input_tour.copy()
    tour_length = len(current_tour)
    current_iteration = 0
    

    while current_iteration < max_iterations:
        infeasible_edges=get_infeasible_edges(current_tour, current_data)
        
        if len(infeasible_edges) == 0:
            # current_tour is feasible
            return current_tour
        
        # Select a random infeasible edge
        infeasible_index = infeasible_edges[np.random.randint(0, len(infeasible_edges))]
        

        # Attempt to find a 2-Opt swap to fix the infeasibility
        swap_found = False
        for k in range(infeasible_index + 2, tour_length - 1):
            if (current_data[current_tour[infeasible_index], current_tour[k]] != np.inf and
                current_data[current_tour[infeasible_index + 1], current_tour[k + 1]] != np.inf):
                
                current_tour = two_opt_swap(current_tour, infeasible_index + 1, k)
                swap_found = True
                break  # Exit the loop after performing a swap
        
        if not swap_found:
            # No feasible swap found for this infeasible edge
            return None
        
        current_iteration += 1
    
    return None
## FINISHED WITH NUMBA FUNCTIONS ## 

## MUTATION FUNCTIONS ##
def mutation_operator(current_candidate, current_type="inversion"):
    if(current_type=="inversion"):
        res=inversion(current_candidate)
        
    if(current_type=="swap"):
        res = swap(current_candidate)
    
    if(res is None):
        print("got none on mutation_operator")
        exit()
    return res
        
def inversion(current_candidate):
    mutated_cand = current_candidate.copy()

    #Getting two random index points in the candidate
    left, right = np.sort(np.random.choice(len(current_candidate) - 1, 2, replace=False))
 
    #Inversion in the 2 points
    mutated_cand[left:right+1]= np.flip(current_candidate[left:right+1])
    mutated_cand[-1]=mutated_cand[0]
    return mutated_cand

def swap(current_candidate):
    mutated_cand = current_candidate.copy()

   #Getting two random index points in the candidate
    i, j = np.random.choice(len(current_candidate) - 1, 2, replace=False)
    
    # Swap the cities
    mutated_cand[i], mutated_cand[j] = mutated_cand[j], mutated_cand[i]
    
    mutated_cand[-1] = mutated_cand[0]
    return mutated_cand

## FINISHED WITH MUTATION FUNCTIONS


## THE OPTIMISER CLASS ## 
#The class for the algorithm:
class gen_algo:

    def __init__(self, pop_size, num_generations, num_parents, alpha, mutation_probability, k, lso_probability, lso_attempts, immigration_factor):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.alpha = alpha
        self.mutation_probability = mutation_probability
        self.k = k
        self.lso_probability = lso_probability
        self.lso_attempts = lso_attempts
        self.immigration_factor=immigration_factor
        print("Running Algorithm with pop_size=%d num_generations=%d num_parents=%d alpha=%.3f mut_probability=%.2f k=%d lso_probability=%.3f lso_attempts=%d immigration_factor=%.3f" % (pop_size, num_generations, num_parents, alpha, mutation_probability, k, lso_probability, lso_attempts, immigration_factor))
    
    def initialise_pop(self):       
        #Num knn of the initial candidates come from the heuristic initialisation, the rest from the random (i.e. two_opt repaired) one
        knn_part, chosen_mut_type=knn_initialisation(self.distanceMatrix,self.pop_size)
        num_kopt=self.pop_size - len(knn_part)
        self.mutation_type = chosen_mut_type
        print("Mutation Type Chosen:", self.mutation_type)
        kopt_part=kopt_initialisation(self.distanceMatrix, num_kopt)
        self.current_population = np.vstack([knn_part, kopt_part])


    ## SELECTION ## 
    #This function takes the current_population, the problem data and the s parameter and returns a selected population,
    #using the quadratic ranking selection.	
    def quadratic_ranking(self, s, replacement=True):	

        #First getting the objective values and doing the ranking
        self.population_objs[:] = [obj_fun(current_cand, self.distanceMatrix) for current_cand in self.current_population]
        ranking=np.argsort(np.argsort(self.population_objs)) + 1

        #Working with the polunomial and getting the pdf
        c=s
        b=len(self.current_population)
        a=(1-s)/((1-b)*(1-b))
        I_scores=a*(ranking-b)*(ranking-b) + c
        pdf=I_scores/np.sum(I_scores)


        #now actually creating the parents	
        indices=np.random.choice(len(self.current_population), size=self.num_parents, replace=replacement, p=pdf)			#for Selection replacement=True, for elimination replacement=False
        
        self.current_parents[:] =self.current_population[indices]
        self.parents_objs[:] = self.population_objs[indices]
    
    def selection(self, current_generation, technique="ranking"):
        if(technique=="ranking"):
            # s=np.pow(self.alpha, current_generation)
            s = self.alpha ** current_generation
            self.quadratic_ranking(s)
    ## FINISHED WITH SELECTION ## 

    ## CROSSOVER ##
    def crossover(self, technique="pmx"):
        
        assert self.num_parents % 2 == 0

        for current_index in range(0, self.num_parents, 2):
            current_parent1=self.current_parents[current_index]
            current_parent2=self.current_parents[current_index+1]
            num_feas_offsprings=0
            while(1):										#Constantly creating offsprings until they are feasible
                offspring1,offspring2=pmx_crossover(current_parent1[:-1], current_parent2[:-1])

                if(not is_feasible(offspring1, self.distanceMatrix)):               #Trying to fix the infeasibility of the candidate
                    repair_result=two_opt_repair(offspring1, self.distanceMatrix)
                    if(repair_result is not None):
                        offspring1=repair_result
                if(not is_feasible(offspring2, self.distanceMatrix)):               #Trying to fix the infeasibility of the candidate
                    repair_result=two_opt_repair(offspring2, self.distanceMatrix)
                    if(repair_result is not None):
                        offspring2=repair_result


                if(is_feasible(offspring1, self.distanceMatrix)):
                    self.current_offspring[current_index+num_feas_offsprings]=offspring1
                    self.offspring_objs[current_index+num_feas_offsprings]=obj_fun(offspring1, self.distanceMatrix)
                    num_feas_offsprings+=1
                    if(num_feas_offsprings>=2):
                        break
                if(is_feasible(offspring2, self.distanceMatrix)):
                    self.current_offspring[current_index+num_feas_offsprings]=offspring2
                    self.offspring_objs[current_index+num_feas_offsprings]=obj_fun(offspring2, self.distanceMatrix)
                    num_feas_offsprings+=1
                    if(num_feas_offsprings>=2):
                        break
    ## FINISHED WITH CROSSOVER

    ## MUTATION ##
    def mutation(self):
        num_candidates=len(self.current_augmented_population)
        for current_index in range(num_candidates):     #Going through any candidate and run test on whether they will be mutated
            if(current_index==0):                   #Basically elitism 
                continue
            random_exp=np.random.random()
            if(random_exp<self.mutation_probability):							#This means that I will apply mutation in this candidate
                current_candidate=self.current_augmented_population[current_index].copy()
                while(1):
                    current_mutation=mutation_operator(current_candidate, self.mutation_type)
                    if(is_feasible(current_mutation, self.distanceMatrix)):
                        self.current_augmented_population[current_index]=current_mutation
                        self.augmented_objs[current_index]=obj_fun(current_mutation, self.distanceMatrix)
                        break
                    else:
                        if(np.random.random() < self.lso_probability):          #If it is not feasible we apply the two_opt_repair operator multiple times for exploration
                            candidates_list=[]
                            objs_list=[]
                            for repaired_candidate in range(self.lso_attempts):                #Applying to the two opt repair process many times 
                                current_repaired=two_opt_repair(current_mutation, self.distanceMatrix)
                                if(current_repaired is not None):               #If the process was succesful, i.e. we got a feasible candidate, append it 
                                    candidates_list.append(current_repaired)
                                    objs_list.append(obj_fun(current_repaired, self.distanceMatrix))
                            if(len(candidates_list) > 0):
                                accepted_mutation=candidates_list[np.argmin(objs_list)]
                                accepted_obj=objs_list[np.argmin(objs_list)]
                                self.current_augmented_population[current_index]=accepted_mutation
                                self.augmented_objs[current_index]=accepted_obj
                                break   
    ## FINISHED WITH MUTATION ##

    ## ELIMINATION ##
    def elimination(self, technique="k_tournament"):

        #Getting the best solution (elitism)
        best_index=np.argmin(self.augmented_objs)
        best_value=self.augmented_objs[best_index]
        best_cand=self.current_augmented_population[best_index].copy()
        #Got the best candidate

        if(technique=="k_tournament"):
            self.k_tournament_selection()

        #Diversity promotion - crowding inspired immigration:
        total_num_immigrants = int (self.immigration_factor * self.pop_size)

        # gettin the indices of every row that appears more than once
        _, first_occurencies=np.unique(self.current_population, axis=0, return_index=True)
        later_occurencies = np.setdiff1d(np.arange(self.pop_size), first_occurencies)  

        #Replacing the values that are observed more than once
        num_immigrants_set=0
        for count,current_index in enumerate(later_occurencies):
            if(num_immigrants_set >= total_num_immigrants):
                break
            current_random_cand=get_kopt_cand(self.distanceMatrix)
            self.current_population[current_index] = current_random_cand
            self.population_objs[current_index] = obj_fun(current_random_cand, self.distanceMatrix)
            num_immigrants_set +=1 

        
        #Now I also add immigrants elsewhere (on random points of the unique points - the 10prcnt best are saved)
        if(num_immigrants_set < total_num_immigrants):              #If this happens i randomly fill the immigrants. The 10 prcnt best solutions are not changed! 
            remaining_immigrants = total_num_immigrants - num_immigrants_set
            sorted_unique_indices = first_occurencies[np.argsort(self.population_objs[first_occurencies])]
            num_unique = len(sorted_unique_indices)
            num_worst  = int(0.9 * num_unique)                          #The amount of the 90 prc worse individuals from the first occurencies
            worst_pool = sorted_unique_indices[-num_worst:]             #The 90prc worse
            indices_to_replace = np.random.choice(worst_pool, size=remaining_immigrants, replace=False)
            for current_index in indices_to_replace:
                current_random_cand=get_kopt_cand(self.distanceMatrix)
                self.current_population[current_index] = current_random_cand
                self.population_objs[current_index] = obj_fun(current_random_cand, self.distanceMatrix)
                num_immigrants_set +=1 

        #Setting the first as the best just 2 be sure
        self.current_population[0]=best_cand
        self.population_objs[0]=best_value





    def k_tournament_selection(self, replacement=False):

        #Parameters and initialising arrays.	
        num_cities=len(self.distanceMatrix)
        seed_pop_size=len(self.current_augmented_population)
        return_pop_size=self.pop_size

        #In each iteration I pick one candidate
        eligibility_flags=np.ones(shape=(seed_pop_size,))		#These will help for the implementation of the replacement=False
        accepted_indices=np.where(eligibility_flags==1)[0]
        for current_iteration in range(return_pop_size):
            if(replacement==False):
                accepted_indices=np.where(eligibility_flags==1)[0]
            sampled_indices= np.random.choice(accepted_indices, size=self.k, replace=False)			#for Selection replacement=True, for elimination replacement=False
            
            #For the sampled candidates I find the best one:
            sampled_candidates=self.current_augmented_population[sampled_indices]
            sampled_candidates_obj=self.augmented_objs[sampled_indices]
            best_cand_index=np.argmin(sampled_candidates_obj)
            best_cand=sampled_candidates[best_cand_index]
            best_cand_obj=sampled_candidates_obj[best_cand_index]


            self.current_population[current_iteration]=best_cand
            self.population_objs[current_iteration]=best_cand_obj


            #I change the accepted indices
            if(replacement==False):
                eligibility_flags[sampled_indices[best_cand_index]]=0
    ## FINISHED WITH ELIMINATION 



    def optimize(self, filename):
        t1=time.time()
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        self.num_cities=len(self.distanceMatrix)
        file.close()
        print("Optimising with filename", filename)
        self.initialise_pop()
        t2=time.time()
        best_vals=[]
        mean_vals=[]

        #Initialising arrays that will be used in the algorithm
        self.current_parents=np.zeros(shape=(self.num_parents, self.num_cities +1), dtype=np.int64)          
        self.current_offspring=np.zeros(shape=(self.num_parents, self.num_cities +1), dtype=np.int64)
        self.current_augmented_population=np.zeros(shape=(self.pop_size + self.num_parents, self.num_cities +1), dtype=np.int64)

        self.population_objs=np.zeros(shape=(len(self.current_population),), dtype=np.float64)
        self.parents_objs=np.zeros(shape=(len(self.current_parents),), dtype=np.float64)
        self.offspring_objs=np.zeros(shape=(len(self.current_offspring),), dtype=np.float64)
        self.augmented_objs=np.zeros(shape=(len(self.current_augmented_population),), dtype=np.float64)

        #Finished initialising the arrays
        for current_generation in range(self.num_generations):
            self.selection(current_generation)
            self.crossover()
            self.current_augmented_population[:self.pop_size]=self.current_population
            self.current_augmented_population[self.pop_size:]=self.current_offspring
            self.augmented_objs[:self.pop_size]=self.population_objs
            self.augmented_objs[self.pop_size:]=self.offspring_objs
            self.mutation()
            self.elimination()

            best_sol_index=np.argmin(self.population_objs)
            best_sol=self.current_population[best_sol_index]
            timeLeft = self.reporter.report(np.mean(self.population_objs), np.min(self.population_objs), best_sol)
            print("Min Obj=%f Mean Obj=%f Gen=%d Time Left=%d" % (np.min(self.population_objs), np.mean(self.population_objs), current_generation, timeLeft))
            if(timeLeft<45):
                break

        #Now applying mass mutation on the best candidate
        best_sol_index=np.argmin(self.population_objs)
        best_sol=self.current_population[best_sol_index]
        best_val=obj_fun(best_sol, self.distanceMatrix)
        print("Now applying mass mutation on the best candidate")
        while(timeLeft>0):
            mutated=mutation_operator(best_sol, self.mutation_type)
            if(not is_feasible(mutated, self.distanceMatrix)):
                rep=two_opt_repair(mutated, self.distanceMatrix)
                if(rep is not None):
                    mutated=rep
            new_obj=obj_fun(mutated, self.distanceMatrix)
            if(new_obj < best_val):
                best_val = new_obj
                best_sol = mutated
                self.current_population[0]=best_sol
                self.population_objs[0]=new_obj
                timeLeft = self.reporter.report(np.mean(self.population_objs), np.min(self.population_objs), best_sol)
                print("Min obj=%f Time left=%d " % (np.min(self.population_objs), timeLeft))
        return 0
    
    
def main():
    
    #Hyperparameters and other info regarding the algorithm:
    pop_size=1600
    num_generations=500
    num_parents=int(2*pop_size)
    alpha=0.995
    mutation_probability=0.25
    k=3                                   #the k of k tournament

    lso_probability = 1
    lso_attempts = 2
    immigration_factor=0.01             #prcntage of population each generation that will be replaced by immigrants after elimination
    
    graph_name="tour1000.csv"
    graph_files_path="data/"
    filename=graph_files_path + graph_name                    #Change according to where you have saved the files in your system 
    #Finished with the hyperparameters
    
    #Initialising the object and running the optimiser function:
    opt_algorithm=gen_algo(pop_size, num_generations, num_parents, alpha, mutation_probability, k, lso_probability, lso_attempts, immigration_factor)
    opt_algorithm.optimize(filename)
 

if __name__=="__main__":
    main()