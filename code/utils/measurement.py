import numpy as np
from scipy import sparse

def public_measurement(public_estimate, marg, weight=1.0):
    """ 
    Take a public estimate and turn it into a measurement tuple
    including a query matrix Q, the estimate, the weight and the marginal
    """
    n = public_estimate.shape[0]
    Q = sparse.eye(n)
    return (Q, public_estimate, weight, marg)

def private_measurement(priv_data, marg, sigma, prng, weight=1.0):
    """
    Take a private dataset answer a marginal query and turn it into a measurement tuple
    including a query matrix Q, the estimate y, the weight and the marginal
    """
    n = priv_data.domain.size(marg)
    Q = sparse.eye(n)
    x = priv_data.project(marg).datavector()
    y = x + gaussian_noise(sigma, n, prng)
    return (Q, y, weight, marg)

def gaussian_noise(sigma, size, prng):
    """ Generate iid Gaussian noise  of a given scale and size """
    return prng.normal(0, sigma, size)