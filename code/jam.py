import os
import json
import pickle
import copy
import numpy as np
import argparse

from mbi import FactoredInference, Dataset, Domain

from utils.accounting import adaptive_split, exponential_mech_eps, gaussian_sigma
from utils.workload import all_k_way, downward_closure, load_from_json
from utils.measurement import private_measurement, public_measurement
from utils.selection import size_filter, exponential_mech, hypothetical_model_size, expected_priv_error
from utils.cdp2adp import cdp_rho


def main(args):
    prng = np.random.RandomState(args.seed)

    datafolder = os.path.join('..',
        'data',
        args.dataset
    )

    domain_path = os.path.join(datafolder, f'{args.dataset}-domain.json')

    pub_data = Dataset.load(path = os.path.join(datafolder, "pub.csv"), domain=domain_path)
    priv_data = Dataset.load(path = os.path.join(datafolder, "priv.csv"), domain=domain_path)
    npub = pub_data.records
    npriv = priv_data.records
    nfactor = npriv / npub

    domain = json.load(open(domain_path))
    domain = Domain(attrs = domain.keys(), shape = domain.values())

    # Workload
    workload = all_k_way(domain.attrs, args.degree)
    
    priv_candidates = downward_closure(workload)
    pub_candidates = downward_closure(workload)
    all_candidates = list(set(priv_candidates).union(set(pub_candidates)))

    unrestricted_priv = copy.deepcopy(priv_candidates)
    unrestricted_pub = copy.deepcopy(pub_candidates)
    unrestricted_all = copy.deepcopy(all_candidates)

    # Get True Answers
    priv_answers = {marg : priv_data.project(marg).datavector() for marg in all_candidates}
    pub_answers = {marg : pub_data.project(marg).datavector() * nfactor for marg in pub_candidates}
    pub_errors = {marg : np.linalg.norm(priv_answers[marg] - pub_answers[marg], 1) for marg in pub_candidates}
    
    # Instanciate engine
    engine = FactoredInference(domain, iters=args.optim_iters)
    model = None
    
    # Convert to zCDP and start tracker for privacy filter
    rho = cdp_rho(args.epsilon, args.delta)
    rho_used = 0
    
    # Sensitivity for neighboring by replacing 1 record
    marg_l2_sensitivity = np.sqrt(2.0)
    score_sensitivity = 4
    
    # Setup tracking
    measurements = []

    # Iterative Selection
    t = 0
    while t < args.rounds:
        increment = 1

        # Get privacy and noise parameters
        rho_select, rho_measure = adaptive_split(rho, rho_used, args.rounds, t, args.alpha)
        selection_eps = exponential_mech_eps(rho_select)
        measurement_sigma = gaussian_sigma(rho_measure, marg_l2_sensitivity)
        
        # Filter By size
        sl = (1/args.rounds) * args.size_limit if t==0 else (rho_used/rho) * args.size_limit
        margs = [measurement[3] for measurement in measurements]
        priv_candidates = [marg for marg in priv_candidates if size_filter(domain, margs, marg, sl)]
        pub_candidates = [marg for marg in pub_candidates if size_filter(domain, margs, marg, sl)]
        all_candidates = list(set(priv_candidates) | set(pub_candidates)) # this elimnates duplicate candidates with set union

        # Get Scores
        if not measurements:
            priv_scores = [-1 * expected_priv_error(measurement_sigma, domain.size(marg)) for marg in priv_candidates]
            pub_scores = [-1 * pub_errors[marg] for marg in pub_candidates]
        else:
            model_error = {marg : np.linalg.norm(priv_answers[marg] - model.project(marg).datavector(), 1) for marg in all_candidates}
            priv_scores = [model_error[marg] - expected_priv_error(measurement_sigma, domain.size(marg)) for marg in priv_candidates]
            pub_scores = [model_error[marg] - pub_errors[marg] for marg in pub_candidates]           

        # Make Selection
        scores = np.array(priv_scores + pub_scores)
        selected_idx = exponential_mech(scores, selection_eps, prng, score_sensitivity)
        rho_used += rho_select

        if not measurements:
            priv_scores = [-1 * expected_priv_error(measurement_sigma, domain.size(marg)) for marg in unrestricted_priv]
            pub_scores = [-1 * pub_errors[marg] for marg in unrestricted_pub]
        else:
            model_error = {marg : np.linalg.norm(priv_answers[marg] - model.project(marg).datavector(), 1) for marg in all_candidates}
            priv_scores = [model_error[marg] - expected_priv_error(measurement_sigma, domain.size(marg)) for marg in priv_candidates]
            pub_scores = [model_error[marg] - pub_errors[marg] for marg in pub_candidates]
        
        scores = np.array(priv_scores + pub_scores)
        unrestricted_idx = exponential_mech(scores, selection_eps, prng, score_sensitivity)
        selected_score, unrestricted_score = scores[selected_idx], scores[unrestricted_idx]
        gap = unrestricted_score - selected_score
        
        # Make Measurement
        if selected_idx < len(priv_scores):
            selected_marg = priv_candidates[selected_idx]
            measurements.append(private_measurement(priv_data, selected_marg, measurement_sigma, prng))
            rho_used += rho_measure
            mtype='priv'
        else:
            shifted_idx = selected_idx - len(priv_scores)
            selected_marg = pub_candidates[shifted_idx]
            measurements.append(public_measurement(pub_answers[selected_marg], selected_marg))
            mtype='pub'

        # Update Model
        print(f'{measurements[-1][-1]} {mtype} {t}/{int(args.rounds)} {measurement_sigma} {selection_eps} {gap}', flush=True)
        model = engine.estimate(measurements, total=npriv)
        size = hypothetical_model_size(domain, model.cliques)

        t += increment

    # Compute Error
    synth = model.synthetic_data()
    errors = []
    for cl in workload:
        X = priv_data.project(cl).datavector()
        Y = synth.project(cl).datavector()
        e = np.linalg.norm(X - Y, 1) * (1 / npriv)
        errors.append(e)
    
    err_avg = np.mean(errors)
    err_max = np.max(errors)

    # Save Results
    results_dir = os.path.join('results', args.dataset, 'jam')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'jam_pgm_{args.dataset}_{args.rounds}.csv'), 'a') as f:
        f.write(f'{args.size_limit},{args.seed},{args.epsilon},{err_avg},{err_max}\n')
    print(f'{args.size_limit},{args.seed},{args.epsilon},{err_avg},{err_max}\n', flush=True)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset Parameters
    parser.add_argument('dataset', type=str, help='dataset name')
    # Privacy Parameters
    parser.add_argument('--epsilon', type=float, default=1.0, help='privacy parameter')
    parser.add_argument('--delta', type=float, default=1e-9, help='privacy parameter')
    parser.add_argument('--alpha', type=float, default=0.2, help='selection budges proportion')
    # Workload Parameters
    parser.add_argument('--degree', type=int, default=3, choices=[1,2,3,4,5], help='degree of marginals in workload')
    # Rng Parameters
    parser.add_argument('--seed', type=int, default=17, help='prng seed')
    # Model Parameters
    parser.add_argument('--optim_iters', type=int, default=1000, help='number of optimization iterations')
    parser.add_argument('--size_limit', type=float, default=80.0, help='model size limit')
    # Mechanism Parameters
    parser.add_argument('--rounds', type=int, default=30, help='number of optimization iterations')

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
