import numpy as np
import itertools
from mbi import Dataset, GraphicalModel, FactoredInference, Domain
from mechanisms.mechanism import Mechanism
from collections import defaultdict
from hdmm.matrix import Identity
from scipy.optimize import bisect
from scipy.stats import mode
import pandas as pd
from mbi import Factor
import argparse

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl)&set(ax)) for ax in workload)
    return { cl : score(cl) for cl in downward_closure(workload) }


def filter_candidates(candidates, model, size_limit):
    ans = { }
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class AIM_pub(Mechanism):
    def __init__(self,epsilon,delta,prng=None,rounds=None,max_model_size=80,alpha=0.1,pub_anneal=False,structural_zeros={}):
        super(AIM_pub, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        self.alpha = alpha
        self.pub_anneal = pub_anneal
        
        
    def worst_approximated(self, candidates, priv_answers, pub_errors, model, eps, sigma, joint_select=True):
        scores = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = priv_answers[cl]
            xest = model.project(cl).datavector()
            current_error = np.linalg.norm(x - xest, 1)
            
            # private candidates
            bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
            scores[(cl, 'priv')] = (current_error - bias) * wgt
            sensitivity[(cl, 'priv')] = 2 * abs(wgt)
            
            # public candidates
            if joint_select:
                scores[(cl, 'pub')] = (current_error - pub_errors[cl]) * wgt
                sensitivity[(cl, 'pub')] = 4 * abs(wgt)
            
        max_sensitivity = max(sensitivity.values()) # if all weights are 0, could be a problem
        return self.exponential_mechanism(scores, eps, max_sensitivity)

    
    def run(self, dpriv, dpub, W):
        rounds = self.rounds or 16*len(dpriv.domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        
        npriv = dpriv.records
        npub = dpub.records
        
        priv_answers = { cl : dpriv.project(cl).datavector() for cl in candidates }
        pub_answers = { cl : dpub.project(cl).datavector() * (npriv / npub) for cl in candidates }
        pub_errors = { cl : np.linalg.norm(priv_answers[cl] - pub_answers[cl], 1) for cl in candidates }
        
        #nmode = int(mode([dv.shape[0] for dv in pub_answers.values()])[0])
        #print('nmode', nmode)
        
        oneway = [cl for cl in candidates if len(cl) == 1]
        
        sigma = np.sqrt(rounds * 2 / (2*(1 - self.alpha)*self.rho)) ## sensitivity^2
        epsilon = np.sqrt(8*self.alpha*self.rho/rounds)
       
        measurements = []
        mtypes = []
        
        zeros = self.structural_zeros
        engine = FactoredInference(dpriv.domain,iters=1000,warm_start=True,structural_zeros=zeros)
        model = engine.estimate(measurements)
        
        print('Initial Sigma', sigma)
        rho_used = len(oneway) * (1.0/8 * epsilon**2) # + 1/sigma**2
        
        #print('Measuring One-way Marginals')
        #for cl in oneway:
        print('First Measurement')
        cl, mtype = self.worst_approximated({c : 1 for c in oneway}, priv_answers, pub_errors, model, epsilon, sigma)
        mtypes.append(mtype)

        n = dpriv.domain.size(cl)
        Q = Identity(n) 

        print('Selected',cl,mtype,'Size',n,'Budget Used',rho_used/self.rho, flush=True)
        if mtype == 'priv':
            rho_used += 1/sigma**2 ## removed 0.5 in numerator
            x = priv_answers[cl]
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, 1.0, cl))  
        elif mtype == 'pub':
            measurements.append((Q, pub_answers[cl], 1.0, cl))
        
        print('The rest') #print('Initialization Complete')
        model = engine.estimate(measurements)
        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2*(1/sigma**2 + 1.0/8 * epsilon**2): ## removed 0.5 in numerator
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(2 / (2*(1 - self.alpha)*remaining)) ## added 2 in numerator
                epsilon = np.sqrt(8*self.alpha*remaining)
                print('(!!!!!!!!!!!!!!!!!!!!!!) Final Round sigma', sigma/2)
                terminate = True
                
            size_limit = self.max_model_size*rho_used/self.rho
            
            small_candidates = filter_candidates(candidates, model, size_limit)
            rho_used += 1.0/8 * epsilon**2
            joint_select = not terminate # don't select public marginals on the final round
            cl, mtype = self.worst_approximated(small_candidates, priv_answers, pub_errors, model, epsilon, sigma, joint_select=joint_select) #
            mtypes.append(mtype)
            
            n = dpriv.domain.size(cl)
            Q = Identity(n) 
            z = model.project(cl).datavector()
            
            if mtype == 'priv':
                rho_used += 1/sigma**2 ## removed 0.5 in numerator
                x = priv_answers[cl]
                y = x + self.gaussian_noise(sigma, n)
                measurements.append((Q, y, 1.0, cl))  
            elif mtype == 'pub':
                measurements.append((Q, pub_answers[cl], 1.0, cl))
            
            model = engine.estimate(measurements)
            w = model.project(cl).datavector()
            print('Selected',cl,mtype,'Size',n,'Budget Used',rho_used/self.rho, flush=True)
            
            if mtype == 'priv' and np.linalg.norm(w-z, 1) <= sigma*np.sqrt(2/np.pi)*n:
                sigma /= 2
                epsilon *= 2
                print('(!!!!!!!!!!!!!!!!!!!!!!) Insufficient Improvement Increasing Budget', sigma)
            elif mtype == 'pub' and self.pub_anneal:
                sigma /= 1.1
                epsilon *= 1.1
                print('(!!!!!!!!!!!!!!!!!!!!!!) Public Measurement Increasing Budget', sigma)
                
        print('Generating Data...')
        engine.iters = 2500
        model = engine.estimate(measurements)
        synth = model.synthetic_data()
        
        return synth, measurements, mtypes