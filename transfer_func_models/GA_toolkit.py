import camb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------ DATA GEN AND SAVE ------------------------------

#obh2vals = np.linspace(0.0214, 0.0234, 4)
#omh2vals = np.linspace(0.13, 0.15, 4)

#make_arr = True
#camb_data = np.empty((4, 0))

#for ombh2_ in obh2vals:
#    for ommh2_ in omh2vals:

#        pars = camb.CAMBparams()
#        H0_=67.5

        #ombh2_ = obh2vals[0]
        #ommh2_ = omh2vals[0]

#        pars.set_cosmology(H0=H0_, ombh2 = ombh2_, omch2 = ommh2_ - ombh2_)
#        pars.WantTransfer = True
#        pars.set_matter_power(redshifts=[0], kmax=10)

#        data = camb.get_transfer_functions(pars)
#        transfer = data.get_matter_transfer_data()
#        kh = transfer.transfer_data[0,:,0]

#        delta = transfer.transfer_data[6,:,0]
#        delta = delta/delta[0] #normalize

#        arr = np.zeros((4, len(kh)))
#        arr[0, :] = ommh2_
#        arr[1, :] = ombh2_
#        arr[2, :] = kh
#        arr[3, :] = delta
#        camb_data = np.concatenate((camb_data, arr), axis=1)

#densities = data.get_background_densities(1.)
#rho_gamma = densities['photon'][0]
#rho_r = densities['photon'][0]# + densities['neutrino'][0] # what else is included in radiation ?
#densities

#np.savetxt("camb_transfer_data.txt", np.transpose(camb_data))
#g = np.loadtxt("camb_transfer_data.txt")
#g = np.transpose(g)


# ------------------------------ CANDIDATE FUNCTION CLASS ------------------------------

class candFunction():
    
    # Candidate Function
    
    def __init__(self):
        
        self.num_a_genes = 19
        self.num_c_genes = 7
        
        self.tot_genes = 19 + 7 # self.num_a_genes + self.num_c_genes
        
        self.params     = np.zeros(self.tot_genes)
        
        self.param_ranges = np.array([[0, 3] for i in range(self.tot_genes)])
        self.param_ranges[13] = [0, 100]
        self.param_ranges[15] = [0, 100]
        self.param_ranges[23] = [0, 100]
        
    def load_params(self, params_file_loc):
        # TODO
        # params = load params from params_file_loc
        # self.set_params(params)
        return 0
    
    def set_params(self, params):
        self.params = params
        
    def set_ranges(self, param_ranges):
        self.param_ranges = param_ranges
    
    def randomize_params(self):#a_rangs: np.ndarray, c_rangs: np.ndarray):
        self.params = np.random.uniform(self.param_ranges[:, 0], self.param_ranges[:, 1])
        
    def mutate(self, ngenes_to_mutate: int = 1, range_mutate: bool = True):
        
        inds = [i for i in range(self.tot_genes)]
        mutate_inds = np.random.choice(inds, ngenes_to_mutate, replace = False)
        
        if range_mutate:
            # Randomize the gene according to the gene's range in a_rangs/c_rangs
            
            for i in mutate_inds:
                self.params[i] = np.random.uniform(self.param_ranges[i, 0], self.param_ranges[i, 1])
                
        else:
            # Multiply the gene by a factor ~1... so as to only slightly modify the gene
            # Use this mutation when the model is expected to be near convergence
            
            # HARD CODED?
            # TUNABLE HYPERPARAMETER?
                # VARIES ACCORDING TO GENERATION ?
                # VARIES BY INPUT VALUE ?
            factor_lb = 0.90 # LOWER BOUND
            factor_ub = 1.10 # UPPER BOUND
            
            for i in mutate_inds:
                
                factor   = np.random.uniform(factor_lb, factor_ub)
                
                self.params[i] *= factor
    
    def compute(self, ommh2, ombh2, k):
        # Equation Numbers refer to https://arxiv.org/abs/2407.16640
        
        num_a_genes = 19
        a_s = self.params[:num_a_genes]
        c_s = self.params[num_a_genes:]
        
        # Eqs. (3) & (4)
        x       = k/(ommh2-ombh2)
        T_nw    = (1+59.0998*x**(1.49177) + 4658.01*x**(4.02755) + 3170.79*x**(6.06) + 150.089*x**(7.28478))**(-0.25)
        
        # Eqs. (17) - (19) w/ trainable a8 - a19
        f_alpha = a_s[7] - a_s[8]*(ombh2**a_s[9]) + a_s[10]*(ommh2**a_s[11])
        f_beta  = a_s[12] - a_s[13]*(ombh2**a_s[14]) + a_s[15]*(ommh2**a_s[16])
        f_node  = a_s[17]*(ommh2**a_s[18])
        
        # Eq. (11)
        s_GA    = (c_s[0]*(ombh2**c_s[1]) + c_s[2]*(ommh2**c_s[3]) + c_s[4]*(ombh2**c_s[5])*(ommh2**c_s[6]))**(-1)
        
        # Eq. (12)
        k_Silk  = 1.6*(ombh2**0.52)*(ommh2**0.73)*(1 + (10.4*ommh2)**(-0.95))
        
        # Eq. (13)
                                     # what to do if this term is negative ?
        f_amp   = f_alpha/(a_s[0] + (f_beta/(k*s_GA))**a_s[1])
        
        # Eq. (14)
        f_Silk  = (k/k_Silk)**a_s[2]
        
        # Eq. (15)
        f_osc   = a_s[3]*k*s_GA/(a_s[4] + f_node/(k*s_GA)**a_s[5])**a_s[6]
        
        # Eq. (5)
        T_w     = 1 + f_amp*np.exp(-f_Silk)*np.sin(f_osc)
        
        Tk      = T_nw * T_w
        return np.real(Tk)
    
# ------------------------------ GENETIC ALGORITHM CLASS ------------------------------

class geneticAlgorithm():
    
    # Fixed functional form, paramters learned
    
    def __init__(self, config):
        
        # compress all input data into a config dict
        self.generations = config["generations"]
        self.population = config["population"]
        self.ranges = config["ranges"]
        
        if type(self.ranges) is np.ndarray:
            self.set_ranges = True
        else:
            self.set_ranges = False
        
        self.kids = np.empty(self.population, dtype=candFunction)
        
        # store the best fitness measure per generation
        self.best_fit_per_gen = np.zeros(self.generations)
        
        self.best_fit_func = candFunction()
        
        self.camb_data = self.load_camb_data(config["camb_data_fname"])
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
            self.kids[i] = candFunction()
            #init_pop[i].set_rangs(self.set_a_rangs, self.set_c_rangs)
            self.kids[i].randomize_params()
            if self.set_ranges:
                self.kids[i].set_ranges(self.ranges)
        
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
        
    
    def load_camb_data(self, camb_data_fname):
        
        data = np.loadtxt(camb_data_fname)
        
        return np.transpose(data)
    
    def compute_fitness(self, candFunc: candFunction):
        
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
        
