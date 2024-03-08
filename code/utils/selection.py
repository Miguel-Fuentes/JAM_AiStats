from mbi import GraphicalModel
import numpy as np
from scipy.special import softmax

l1_gauss_factor = np.sqrt(2 / np.pi)

def expected_priv_error(sigma, n):
    """ Compute the expected error of a private estimate computed via the Gaussian mechanism with a given sigma and marg size n"""
    return sigma * n * l1_gauss_factor

def hypothetical_model_size(domain, cliques):
    """ Compute the size of a hypothetical model (in MB) with a given domain and cliques """
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20

def size_filter(domain, margs, marg, size_limit):
    """ Check if a marginal can be added to a model without exceeding the size limit """
    return hypothetical_model_size(domain, margs + [marg]) <= size_limit

def workload_size_filter(domain, model, marg, workload, size_limit):
    baseline = model.cliques + [marg] if model else [marg]
    return all(hypothetical_model_size(domain, baseline + [cl]) <= size_limit for cl in workload)
    
def exponential_mech(scores, epsilon, prng, sensitivity):
    """ Sample from the exponential mechanism with scores, epsilon, and sensitivity """
    probs = softmax((epsilon*scores) / (sensitivity * 2))
    num_candidates = len(scores)
    return prng.choice(num_candidates, 1, p=probs)[0]