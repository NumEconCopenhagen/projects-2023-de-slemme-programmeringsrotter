from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = np.minimum(HM, HF)
        elif par.sigma == 1:
            H = (HM**(1-par.alpha)*HF**par.alpha)
        else:
            HM=np.fmax(HM, 1e-07)
            HF=np.fmax(HF, 1e-07)
            inner = (1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma)
            H=np.fmax(inner, 1e-07) **(par.sigma/(par.sigme-1))
        #Løsning for sigma = 1. Lav H om til et if statement

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-7)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        # define parameters and namespaces
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()  
        
        # constraints and bounds; 
        # constraints define that e.g. male cannot have the sum of work hours and household hours be more than 24.
        # bounds define that one cannot work more than 24 in either household or work hours
        constraints_m = ({'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[1]}) 
        constraints_f = ({'type': 'ineq', 'fun': lambda x: 24 - x[2] - x[3]}) 
        constraints = [constraints_m, constraints_f]
        bounds = [(0,24)]*4 

        # print bounds for additional info; output seems correct
       # print("Bound test:")
        #print(f'Bounds for [LM, HM, LF, HF]: {bounds}\n')

        # call optimizer
        initial_guess = [10]*4
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])
        res = optimize.minimize(obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol = 0.000000001) #Hvis tolerancen er høj, accepterer den løsninger som kun er tæt på at være rigtige
        
        # save results into opt values
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        # print out the values of opt
      #  print("Optimal choices:")
     #   if do_print:
      #      for k,v in opt.__dict__.items():
       #         print(f'{k} = {v:6.4f}')

        return opt
    

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        sol = self.sol
        par = self.par

        for i, w_F in enumerate(par.wF_vec):
            par.wF = w_F
            if discrete:
                opt = self.solve_discrete()
            else:
                opt = self.solve()
            if opt is not None:
                sol.LM_vec[i], sol.HM_vec[i], sol.LF_vec[i], sol.HF_vec[i] = opt.LM, opt.HM, opt.LF, opt.HF
                
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        beta0_hat_list = []
        beta1_hat_list = []
        
        
        
class HouseholdSpecializationModelClassExtended(HouseholdSpecializationModelClass):

    def estimateV2(self, alpha=None, sigma=None):
        """ estimate alpha and sigma """
        
        par = self.par
        sol = self.sol

        # define objective function to minimize
        def objective(x):
            alpha, sigma = x
            par.alpha = alpha
            par.sigma = sigma
            self.solve_wF_vec()
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
        
        # initial guess
        guess = [0.5, 1]

        # call solver
        solution = optimize.minimize(objective, guess, method='Nelder-Mead')

        alpha, sigma = solution.x

        return alpha, sigma


    def estimateV3(self, sigma=None):
        """ estimate alpha and sigma """
        
        par = self.par
        sol = self.sol

        # define objective function to minimize
        def objective(x):
            par.sigma = x[0]
            self.solve_wF_vec()
            self.run_regression()
            return (0.4 - sol.beta0)**2 + (-0.1 - sol.beta1)**2

        guess = [1.4]
        bounds = [(0, 30)]

        result = optimize.minimize(objective, guess, method='Nelder-Mead', bounds=bounds)

        sigma = result.x[0]

        return sigma
