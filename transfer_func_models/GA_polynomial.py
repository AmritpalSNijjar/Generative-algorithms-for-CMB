import camb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------ CANDIDATE POLYNOMIAL FUNCTION CLASS ------------------------------


class candPolynomial():
    
    # Candidate Polynomial Function
    
    def __init__(self, var_names, order = "first", learn_offset = True):
        
        # candPolynomial is of the form: 
        # offset + A1 * var_1 ^ B1              + A2 * var_2 ^ B2              + .... 
        #        + C1 * var_1 ^ D1 * var_2 ^ D2 + C2 * var_1 ^ D3 * var_3 ^ D4 + ...,
        
        # Where var_1 : var_names[0], var_2 : var_names[1], var_3 : var_names[2], ...
        
        # - offset may be learnable or permanently set at 0
        
        self.learn_offset         = learn_offset
        self.order                = order
        self.var_names            = var_names
        self.n_vars               = len(self.var_names)
        
        self.offset               = 0
        
        self.n_first_order_terms  = self.n_vars
        self.first_order_keeps    = np.ones(self.n_first_order_terms, dtype = int)
        self.first_order_coeffs   = np.ones(self.n_first_order_terms)     # [A1, A2, ...]
        self.first_order_exps     = np.zeros(self.n_first_order_terms)    # [B1, B2, ...]
        
        self.n_second_order_terms = (self.n_vars*(self.n_vars - 1))//2    # n(n-1)/2
        self.second_order_keeps   = np.ones(self.n_second_order_terms, dtype = int)
        self.second_order_coeffs  = np.ones(self.n_second_order_terms)    # [C1, C2, ...]
        self.second_order_exps    = np.zeros(2 * self.n_second_order_terms)  # [D1, D2, D3, D4, ...]
        
        self.second_order_orders  = np.zeros((self.n_second_order_terms, 2), dtype = int)
        
        a = 0
        b = 1
        for i in range(self.n_second_order_terms):
            
            self.second_order_orders[i, 0] = a
            self.second_order_orders[i, 1] = b
            
            if b == self.n_vars - 1:
                    a += 1
                    b  = a + 1
                else:
                    b += 1
                    
        self.offset_range              = np.array([-10, 10]) 
        
        self.first_order_coeff_ranges  = np.array([[-10, 10] for i in range(self.n_first_order_terms)])
        self.first_order_exp_ranges    = np.array([[0, 3] for i in range(self.n_first_order_terms)])
        
        self.second_order_coeff_ranges = np.array([[-2, 2] for i in range(self.n_second_order_terms)])
        self.second_order_exp_ranges   = np.array([[0, 3] for i in range(2*self.n_second_order_terms)])
    
    def set_params(self, params_dict):
        
        self.offset = params_dict["offset"]
        
        self.first_order_keeps   = params_dict["first_order_keeps"]
        self.first_order_coeffs  = params_dict["first_order_coeffs"]
        self.first_order_exps    = params_dict["first_order_exps"]
        
        self.second_order_keeps  = params_dict["second_order_keeps"]
        self.second_order_coeffs = params_dict["second_order_coeffs"]
        self.second_order_exps   = params_dict["second_order_exps"]
        
    def set_ranges(self, ranges_dict):
        
        self.offset_range               = ranges_dict["offset_range"]
        
        self.first_order_coeff_ranges   = ranges_dict["first_order_coeff_ranges"]
        self.first_order_exp_ranges     = ranges_dict["first_order_exp_ranges"]
        
        self.second_order_coeff_ranges  = ranges_dict["second_order_coeff_ranges"]
        self.second_order_exp_ranges    = ranges_dict["second_order_exp_ranges"]
    
    def randomize_params(self, rand_keep_polys = True):
        
        self.first_order_coeffs  = np.random.uniform(self.first_order_coeff_ranges[:, 0], self.first_order_coeff_ranges[:, 1])
        self.first_order_exps    = np.random.uniform(self.first_order_exp_ranges[:, 0], self.first_order_exp_ranges[:, 1])
        
        self.second_order_coeffs = np.random.uniform(self.second_order_coeff_ranges[:, 0], self.second_order_coeff_ranges[:, 1])
        self.second_order_exps   = np.random.uniform(self.second_order_exp_ranges[:, 0], self.second_order_exp_ranges[:, 1])
        
        if self.learn_offset:
            self.offset          = np.random.uniform(self.offset_range[0], self.offset_range[1])
        
        if rand_keep_polys:
            self.first_order_keeps  = np.random.choice([0, 1], self.n_first_order_terms)
            self.second_order_keeps = np.random.choice([0, 1], self.n_second_order_terms)
    
    def mutate_form(self, nterms_to_flip: int = 1, order_to_flip = "all"):
        
        # if first/second_order_keeps is something like [1, 1, 0, 1, 0],
        
        # this could be mutated to any of the following:
        #                           - [1, 0, 0, 1, 0]
        #                             [1, 1, 0, 0, 0]
        #                             [0, 1, 0, 1, 0]
        #                             [1, 1, 0, 1, 1]
        #                             [1, 1, 1, 1, 0] etc. etc.
        
        n1 = self.n_first_order_terms
        first_order_inds = [i for i in range(n1)] 
        
        if self.order == "second":
            n2 = self.n_second_order_terms
            second_order_inds = [j + n1 for j in range(n2)]
        
        if order_to_flip == "all":
            inds = first_order_inds + second_order_inds
        elif order_to_flip == "first":
            inds = first_order_inds
        elif order_to_flip == "second":
            inds = seond_order_inds
        
        mutate_inds = np.random.choice(inds, nterms_to_flip, replace = False)
        
        for i in mutate_inds:
            
            if i >= n1:
                j = i - n1
                self.second_order_keeps[j] = 1 - self.second_order_keeps[j]  # 1 -> 0 or 0 -> 1
            else:
                self.first_order_keeps[i]  = 1 - self.first_order_keeps[i]
    
    def mutate_params(self, ngenes_to_mutate: int = 1):
        
        # only terms which are marked as KEEP in keep_poly will be available to mutate
        n1 = self.n_first_order_terms
        first_coeff_inds = [i for i in range(n1) if self.first_order_keeps[i] == 1]            # [0,..., n1 - 1 ], at most
        first_exp_inds = [i + n1 for i in range(n1) if self.first_order_keeps[i] == 1]         # [n1,..., 2*n1 - 1]
        inds = first_coeff_inds + first_exp_inds
        
        n2 = self.n_second_order_terms
        
        if self.order == "second:"
            second_coeff_inds = [j + 2*n1 for j in range(n2) if self.second_order_keeps[i] == 1]   # [2*n1,..., 2*n1 + n2 - 1]
            second_exp_inds = [j + 2*n1 + n2 for j in range(n2) if self.second_order_keeps[i] == 1]# [2*n1 + n2,..., 2*n1 + 2*n2 - 1]
            inds = inds + second_coeff_inds + second_exp_inds
        
        if self.learn_offset:
            inds.append(inds[-1] + 1)
        
        mutate_inds = np.random.choice(inds, ngenes_to_mutate, replace = False)
        
        for i in mutate_inds:
            
            if self.learn_offset and i == inds[-1]:
                self.offset = np.random.uniform(self.offset_range[0], self.offset_range[1])

            if i >= 2*n1 + n2:
                j = i - (2*n1 + n2) 
                self.second_order_exps[j] = np.random.uniform(self.second_order_exp_ranges[j, 0], self.second_order_exp_ranges[j, 1])
            elif i >= 2*n1:
                j = i - 2*n1
                self.second_order_coeffs[j] = np.random.uniform(self.second_order_coeff_ranges[j, 0], self.second_order_coeff_ranges[j, 1])
            elif i >= n1:
                j = i - n1
                self.first_order_exps[j] = np.random.uniform(self.first_order_exp_ranges[j, 0], self.first_order_exp_ranges[j, 1])
            else:
                self.first_order_coeffs[j] = np.random.uniform(self.first_order_coeff_ranges[j, 0], self.first_order_coeff_ranges[j, 1])
    
    def compute(self, vars_dict):
        
        # vars_dict looks like {"omega_b": 0.0224, "omega_m": 0.315, ...}
        
        val = self.offset
        
        for i in range(self.n_first_order_terms):
            val += self.first_order_keeps[i] * self.first_order_coeffs[i] * (vars_dict[self.var_names[i]] ** self.first_order_exps[i])
            
        if self.order == "second":
            
            for i in range(self.n_second_order_terms):
                
                val += self.second_order_keeps[i] * self.second_order_coeffs[i] * (vars_dict[self.var_names[self.second_order_orders[i, 0]]] ** self.second_order_coeffs[2*i]) * (vars_dict[self.var_names[self.second_order_orders[i, 1]]] ** self.second_order_coeffs[2*i + 1])
        
        return val

# ------------------------------ DATA POINT CLASS ------------------------------
    
    
class dataPoint():
    def __init__(self, params_dict, x_var, y_var):
        
        self.params_dict = params_dict # could have any params, including ones not being modelled by GA. but make sure the 
                                       # string names of the params used in GA polynomials are the same strings here
        self.x_var = x_var # x-variable to make the plot we want to model
        self.y_var = y_var # y-variable values of the true (simulated) plot

# ------------------------------ ALGORITHM FUNCTION CLASS ------------------------------

class geneticAlgorithmPolynomial():
    
    # Polynomial form..... 
    
    def __init__(self, config, data_list):
        
        # compress all input data into a config dict
        self.generations = config["generations"]
        self.population = config["population"]
        self.ranges_dict = config["ranges_dict"]
        
        self.train_data = data_list # this should be a LIST of dataPoint instances
        
        # cand polynomial requirements
        self.var_names = config["var_names"]
        self.order = config["order"]
        self.learn_offset = config["learn_offset"]
        
        self.kids = np.empty(self.population, dtype=candPolynomial)
        
        # store the best fitness measure per generation
        self.best_fit_per_gen = np.zeros(self.generations)
        
        self.best_fit_func = candPolynomial(var_names = self.var_names, order = self.order, learn_offset = self.learn_offset)
        
        #self.camb_data = self.load_camb_data(config["camb_data_fname"])
        # [[ommh2], [ombh2], [k], [T(k)]]
        
        self.mutation_rate = config["mutation_rate"]
        
        # Percent best kids to keep i.e. keep the 40% most fit kids
        # Need to play around with this value
        self.percent_winners = 0.40
        
        # NUMBER OF GENES TO MUTATE PER POP
        # Need to play around with this value
        self.ngenes_to_mutate = 2 
        
        self.num_winners = int(self.percent_winners*self.population)
        
        self.num_losers  = self.population - self.num_winners
        
    
    def init_pop(self):
        print("Initializing random initial population...")
        for i in range(self.population):
            self.kids[i] = candPolynomial(var_names = self.var_names, order = self.order, learn_offset = self.learn_offset)
            self.kids[i].randomize_params()
            if self.set_ranges:
                self.kids[i].set_ranges(self.ranges_dict)
        
    def run_algorithm(self):
        
        # generate random initial population
        self.init_pop()
        
        
        print("Starting genetic algorithm...")
        for i in range(self.generations):
            
            self.sort_winners(i)
            if i%10 == 0:
                print(f"Generation: {i + 1}")
                print(f"Most fit: {self.best_fit_per_gen[i]}")
            
            # a & b denote the indices of the parent functions to breed kid functions from
            # 0 is the most fit function, 1 is the second most fit, etc.
            # the current algorithm works as follows: if there were possible parent functions 0, 1, 2, 3, 4
            
            # kids1, kids2 = crossbreed(parent0, parent1)
            # kids3, kids4 = crossbreed(parent0, parent2)
            # ...
            # kids7, kids8 = crossbreed(parent0, parent4)
            # kids9, kids10 = crossbreed(parent1, parent2)
            # ...
            
            
            a = 0
            b = 1
            
            for i in range(self.num_winners, self.population, 2):
            
                if i + 1 <= self.population - 1:
                    self.kids[i], self.kids[i + 1] = self.crossbreed(self.kids[a], self.kids[b])
                else: #(i + 1 is invalid entry... only want 1 kid)
                    self.kids[i], _ = self.crossbreed(self.kids[a], self.kids[b])
                    
                if b < self.num_winners - 1:
                    b += 1
                else:
                    a += 1
                    b = a + 1
            
            # TO DO: Dynamic mutation rate, dynamic ngenes_to_mutate, range_mutate turned off eventually
            
            for i in range(0, self.population):
                if np.random.uniform() <= self.mutation_rate:
                    self.kids[i].mutate(ngenes_to_mutate = self.ngenes_to_mutate, range_mutate = True)
        
        
        self.sort_winners(i)
        self.best_fit_func = self.kids[0]
    
    def compute_fitness(self, candPoly: candPolynomial):
        
        fitness = 0
        
        #TO DO: For quick and rough fitness computation, for example if num_points is large, we can
        #       instead sample a number of points from the data instead of using it all...
        
        num_points = self.camb_data.shape[1]
        
        # camb_data file will look like:
        # [[ommh2], [ombh2], [k], [T(k)]]
        
        for i in range(num_points):
            val = np.abs(self.camb_data[3][i] - candFunc.compute(self.camb_data[0][i], self.camb_data[1][i], self.camb_data[2][i]))
            val /= self.camb_data[3][i]
            fitness += val
        
        # Eq. (16)
        fitness *= 100/num_points
        return fitness
    
    def save_top_func(self):
        # TO DO: Auto save the best function... text file with just params?
        # best_function = self.kids[-1, 0]
        # pickle save:
        # As = best_function.getAs
        # Cs = best_function.getCs
        return 0
    
    def sort_winners(self, cur_gen):
        
        # sort funcions by fitness... most fit goes first, then second, etc
        
        fitnesses = [self.compute_fitness(kid) for kid in self.kids]
        inds = np.argsort(fitnesses)
        self.best_fit_per_gen[cur_gen] = fitnesses[inds[0]]
        sorted_kids = [self.kids[inds[i]] for i in range(self.population)]
        self.kids = np.array(sorted_kids)
    
    def crossbreed(self, parent1: candFunction, parent2: candFunction) -> tuple[candFunction, candFunction]:
    
        par1_genes = parent1.params
        par2_genes = parent2.params


        # This part can be reworked to have random genes in the genome swap instead of a 
        # 50/50 split down the middle
        genes_swap_length = parent1.tot_genes//2

        kid1_genes = np.concatenate((par1_genes[:genes_swap_length], par2_genes[genes_swap_length:]))
        kid2_genes = np.concatenate((par2_genes[:genes_swap_length], par1_genes[genes_swap_length:]))
        # --------------------------------------------------

        kid1 = candFunction()
        kid2 = candFunction()
        kid1.set_params(kid1_genes)
        kid2.set_params(kid2_genes)
        
        if self.set_ranges:
            kid1.set_ranges(self.ranges)
            kid2.set_ranges(self.ranges)
            
        return kid1, kid2
