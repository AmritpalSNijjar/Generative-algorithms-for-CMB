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
    
    def represent(self):
        representation = f"p = {self.offset} +"
        for i in range(self.n_first_order_terms):
            if self.first_order_keeps[i] == 1:
                representation += f" {self.first_order_coeffs[i]}*{self.var_names[i]}^{self.first_order_exps[i]} +"
        for i in range(self.n_second_order_terms):
            if self.second_order_keeps[i] == 1:
                representation += f" {self.second_order_coeffs[i]}*({self.var_names[self.second_order_orders[i, 0]]}^{self.second_order_exps[2*i]})*({self.var_names[self.second_order_orders[i, 1]]}^{self.second_order_exps[2*i+1]}) +"
        representation = representation[:-1] + "."
        print(representation)
    
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
        
        if self.order == "second":
            second_coeff_inds = [j + 2*n1 for j in range(n2) if self.second_order_keeps[j] == 1]   # [2*n1,..., 2*n1 + n2 - 1]
            second_exp_inds = [j + 2*n1 + n2 for j in range(n2) if self.second_order_keeps[j] == 1]# [2*n1 + n2,..., 2*n1 + 2*n2 - 1]
            inds = inds + second_coeff_inds + second_exp_inds
            
        
        if self.learn_offset:
            if len(inds) == 0:
                inds = [0]
            else:
                inds.append(inds[-1] + 1)
        
        mutate_inds = np.random.choice(inds, min(ngenes_to_mutate, len(inds)), replace = False)
        
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
                self.first_order_coeffs[i] = np.random.uniform(self.first_order_coeff_ranges[i, 0], self.first_order_coeff_ranges[i, 1])
    
    def compute(self, vars_dict):
        
        # vars_dict looks like {"omega_b": 0.0224, "omega_m": 0.315, ...}
        
        val = self.offset
        
        for i in range(self.n_first_order_terms):
            val += self.first_order_keeps[i] * self.first_order_coeffs[i] * (vars_dict[self.var_names[i]] ** self.first_order_exps[i])
            
        if self.order == "second":
            
            for i in range(self.n_second_order_terms):
                
                val += self.second_order_keeps[i] * self.second_order_coeffs[i] * (vars_dict[self.var_names[self.second_order_orders[i, 0]]] ** self.second_order_exps[2*i]) * (vars_dict[self.var_names[self.second_order_orders[i, 1]]] ** self.second_order_exps[2*i + 1])
        
        return val

# ------------------------------ DATA POINT CLASS ------------------------------
    
    
class dataPoint():
    def __init__(self, params_dict, x_var, y_var):
        
        self.params_dict = params_dict # could have any params, including ones not being modelled by GA. but make sure the 
                                       # string names of the params used in GA polynomials are the same strings here
            
            
        # (x_var, y_var) ARE THE POINTS OF WHAT'S TO BE MODELLED
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
        
        self.t_subgroup_size = config["t_subgroup_size"]
        self.t_win_probability = config["t_win_probability"]
        
        self.train_data = data_list # this should be a LIST of dataPoint instances
        self.n_data = len(self.train_data)
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
        
        self.mutation_rates = config["mutation_rate"] #[% total mutation rate, % mutate params]
        
        # Percent best kids to keep i.e. keep the 40% most fit kids
        # Need to play around with this value
        self.percent_keep_per_gen = config["percent_keep_per_gen"]
        
        # NUMBER OF GENES TO MUTATE PER POP
        # Need to play around with this value
        self.ngenes_to_mutate = 2 
        
        self.num_winners = int(self.percent_keep_per_gen*self.population)
        
        self.num_losers  = self.population - self.num_winners
        
    
    def init_pop(self):
        print("Initializing random initial population...")
        for i in range(self.population):
            self.kids[i] = candPolynomial(var_names = self.var_names, order = self.order, learn_offset = self.learn_offset)
            self.kids[i].randomize_params()
            
            #if self.set_ranges:
            #    self.kids[i].set_ranges(self.ranges_dict)
        
    def run_algorithm(self):
        
        # generate random initial population
        self.init_pop()
        print("Starting genetic algorithm...")
        print("Generation --    Fitness")
        for i in range(self.generations):
            
            self.tournament(i)
            if i%10 == 0:
                print(f"{i + 1} --    {self.best_fit_per_gen[i]}")
            
            # a & b denote the indices of the parent functions inds from where to breed kid functions from
            
            potential_parent_inds = [i for i in range(self.num_winners)]
            
            for j in range(self.num_winners, self.population, 2):
                
                [a, b] = np.random.choice(potential_parent_inds, size = 2,replace = False)
                
                if j + 1 <= self.population - 1:
                    self.kids[j], self.kids[j + 1] = self.crossbreed(self.kids[a], self.kids[b])
                else: #(i + 1 is invalid entry... only want 1 kid)
                    self.kids[j], _ = self.crossbreed(self.kids[a], self.kids[b])
            
            # TO DO: Dynamic mutation rate, dynamic ngenes_to_mutate, range_mutate turned off eventually
            
                
            for k in range(0, self.population):
                if np.random.uniform() <= self.mutation_rates[0]:
                    if np.random.uniform() <= self.mutation_rates[1]:
                        self.kids[k].mutate_params(ngenes_to_mutate = 3)
                    else:
                        self.kids[k].mutate_form(nterms_to_flip = 1)
        
        
        self.tournament(i)
        final_fitnesses = [self.compute_fitness(kid) for kid in self.kids]
        most_fit_ind = np.argmin(final_fitnesses)
        self.best_fit_func = self.kids[most_fit_ind]
    
    def compute_fitness(self, candPoly: candPolynomial):
        
        #TO DO: For quick and rough fitness computation, for example if num_points is large, we can
        #       instead sample a number of points from the data instead of using it all...
        
        N = 0
        
        # BATCH TRAIN:
        # batch_size = self.n_data//3
        # dat_inds = np.random.choice([i for i in range(self.n_data)], size = batch_size, replace = False)
        
        # TRAIN ON ALL DATA:
        dat_inds = [i for i in range(self.n_data)]
        
        for ind in dat_inds:
            
            dat = self.train_data[ind]
            
            polynomial_eval = candPoly.compute(dat.params_dict)
            
            eval_at_points  = dat.x_var
            N              += len(eval_at_points)
            truth           = dat.y_var

            # modeled_f IS HARD-CODED... WHERE DOES THE POLYNOMIAL COME INTO THE FUNCTIONAL FORM OF THE MODEL
            # in this case..... our modeled func is:
            # f(k; a, b, ...) = k ^ p(a, b, ...)  where p(a, b, ...) would be some polynomial of parameters a, b, ...

            modeled_f = eval_at_points**(polynomial_eval) 
            # ^(TODO: MAKE MODELED_F A FUNCTION THAT IS FED INTO THE GA THROUGH THE CONFIG FILE...... LIKE SELF.MODEL_SKELETON) 

            val = np.abs((modeled_f - truth)/(truth+1e-12))
            
            fitness = np.sum(val)
        
        fitness *= 100/N
        
        return fitness
    
    def tournament(self, cur_gen):
        
        #k: TOURNAMENT SUBGROUP SIZE
        k = self.t_subgroup_size
        
        #p: SUBGROUP SAMPLING PROPBABILITY
        p = self.t_win_probability
        
        #create temporary array to store tournament winners
        tournament_winners = np.empty(self.num_winners, dtype=candPolynomial)
        
        inds = [i for i in range(self.population)]
        
        t_best_fitness = 0
        
        for i in range(self.num_winners):
            # Tournament round
            
            subgroup_inds = np.random.choice(inds, k, replace = False)
            
            subgroup_fitnesses = [self.compute_fitness(self.kids[ind]) for ind in subgroup_inds]
            
            subgroup_argsort = np.argsort(subgroup_fitnesses)
            subgroup_fitness_rankings = np.argsort(subgroup_argsort)
            
            probabilities = p*(1-p)**np.minimum(subgroup_fitness_rankings, np.zeros_like(subgroup_inds)+7)
            
            
            # To make sure probabilities array sums to 1
            diff_from_one = 1 - np.sum(probabilities)
            probabilities[np.argmax(probabilities)] += diff_from_one
            
            winner_ind = np.random.choice(subgroup_inds, size = None, p = probabilities)
            
            tournament_winners[i] = self.kids[winner_ind]
            
            winner_ind_in_subgroup = np.where(subgroup_inds == winner_ind)[0][0]
            winner_fitness = subgroup_fitnesses[winner_ind_in_subgroup]
            
            if i == 0: 
                t_best_fitness = winner_fitness
            
            t_best_fitness = min(t_best_fitness, winner_fitness)
        
        for i in range(self.num_winners):
            self.kids[i] = tournament_winners[i]
        
        self.best_fit_per_gen[cur_gen] = t_best_fitness
    
    
    def sort_winners(self, cur_gen):
        
        # sort funcions by fitness... most fit goes first, then second, etc
        
        fitnesses = [self.compute_fitness(kid) for kid in self.kids]
        inds = np.argsort(fitnesses)
        self.best_fit_per_gen[cur_gen] = fitnesses[inds[0]]
        sorted_kids = [self.kids[inds[i]] for i in range(self.population)]
        self.kids = np.array(sorted_kids)
    
    def crossbreed(self, parent1: candPolynomial, parent2: candPolynomial):
        
        par1_first_order_coeffs  = parent1.first_order_coeffs
        par1_first_order_exps    = parent1.first_order_exps
        par1_second_order_coeffs = parent1.second_order_coeffs
        par1_second_order_exps   = parent1.second_order_exps
        
        par2_first_order_coeffs  = parent2.first_order_coeffs
        par2_first_order_exps    = parent2.first_order_exps
        par2_second_order_coeffs = parent2.second_order_coeffs
        par2_second_order_exps   = parent2.second_order_exps
        
        kid1_params_dict = {}
        kid2_params_dict = {}
        
        L = len(parent1.first_order_keeps)//2
        kid1_params_dict["first_order_keeps"]   = np.concatenate((parent1.first_order_keeps[:L], parent2.first_order_keeps[L:]))
        kid2_params_dict["first_order_keeps"]   = np.concatenate((parent2.first_order_keeps[:L], parent1.first_order_keeps[L:]))                                                       
        kid1_params_dict["second_order_keeps"]  = np.concatenate((parent2.second_order_keeps[:L], parent1.second_order_keeps[L:]))
        kid2_params_dict["second_order_keeps"]  = np.concatenate((parent1.second_order_keeps[:L], parent2.second_order_keeps[L:]))
        
        kid1_params_dict["offset"]              = parent2.offset
        kid2_params_dict["offset"]              = parent1.offset
        
        L = len(par1_first_order_coeffs)//2
        kid1_params_dict["first_order_coeffs"]  = np.concatenate((par1_first_order_coeffs[:L], par2_first_order_coeffs[L:]))
        kid2_params_dict["first_order_coeffs"]  = np.concatenate((par2_first_order_coeffs[:L], par1_first_order_coeffs[L:]))
        
        L = len(par1_first_order_exps)//2
        kid1_params_dict["first_order_exps"]    = np.concatenate((par1_first_order_exps[:L], par2_first_order_exps[L:]))
        kid2_params_dict["first_order_exps"]    = np.concatenate((par2_first_order_exps[:L], par1_first_order_exps[L:]))
        
        L = len(par1_second_order_coeffs)//2
        kid1_params_dict["second_order_coeffs"] = np.concatenate((par1_second_order_coeffs[:L], par2_second_order_coeffs[L:]))
        kid2_params_dict["second_order_coeffs"] = np.concatenate((par2_second_order_coeffs[:L], par1_second_order_coeffs[L:]))
        
        L = len(par1_second_order_exps)//2
        kid1_params_dict["second_order_exps"]   = np.concatenate((par1_second_order_exps[:L], par2_second_order_exps[L:]))
        kid2_params_dict["second_order_exps"]   = np.concatenate((par2_second_order_exps[:L], par1_second_order_exps[L:]))

        kid1 = candPolynomial(self.var_names, order = self.order, learn_offset = self.learn_offset)
        kid2 = candPolynomial(self.var_names, order = self.order, learn_offset = self.learn_offset)
        kid1.set_params(kid1_params_dict)
        kid2.set_params(kid2_params_dict)
        
        #if self.set_ranges:
        #    kid1.set_ranges(self.ranges)
        #    kid2.set_ranges(self.ranges)
            
        return kid1, kid2
