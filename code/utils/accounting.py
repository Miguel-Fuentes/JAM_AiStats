import numpy as np

from utils.cdp2adp import cdp_rho

def adaptive_split(rho, rho_used, T, t, alpha):
    """
    Adaptively split the rho into a rho for selection and a rho for measurement based on an alpha, 
    the remaining budget rho_used and the total budget rho, the index of the current round t and the total number of rounds T.
    """
    rho_t = (rho - rho_used) / (T - t)

    rho_select = alpha * rho_t
    rho_measure = (1 - alpha) * rho_t
    
    return rho_select, rho_measure

def exponential_mech_eps(rho):
    """Compute the epsilon for the exponential mechanism for a given zCDP-rho """
    return np.sqrt(8 * rho)

def gaussian_sigma(rho, l2_sensitivity):
    """ Compute the Gaussian noise scale for a given zCDP-rho and l2_sensitivity """
    sigma2 = (l2_sensitivity**2) / (2 * rho)
    sigma = np.sqrt(sigma2)
    return sigma
