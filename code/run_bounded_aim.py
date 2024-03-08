from mechanisms.bounded_aim import AIM

import os
import pandas as pd
import numpy as np
import itertools
import argparse
import json

from mbi import Dataset, Domain
from utils.workload import all_k_way, downward_closure

def main(args):
    datafolder = os.path.join('..',
        'data',
        args.dataset
    )

    domain_path = os.path.join(datafolder, f'{args.dataset}-domain.json')

    priv_data = Dataset.load(path = os.path.join(datafolder, "priv.csv"), domain=domain_path)
    npriv = priv_data.records

    domain = json.load(open(domain_path))
    domain = Domain(attrs = domain.keys(), shape = domain.values())
    
    # Workload
    workload = all_k_way(domain.attrs, args.degree)
    
    modified_workload = [(cl, None) for cl in workload]
    rng = np.random.default_rng(args.seed)
    
    algo = AIM(args.epsilon,
           args.delta,
           prng=rng,
           max_model_size=args.size_limit,
           structural_zeros={})
    
    data = algo.run(priv_data, modified_workload)
    
    errors = []
    for cl in workload:
        X = priv_data.project(cl).datavector()
        Y = data.project(cl).datavector()
        e = np.linalg.norm(X - Y, 1) * (1 / npriv)
        errors.append(e)
        
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    
    # Save Results
    results_dir = os.path.join('results', args.dataset, 'aim')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'aim_{args.dataset}.csv'), 'a') as f:
        f.write(f'{args.size_limit},{args.seed},{args.epsilon},{mean_err},{max_err}\n')
    print(f'{args.size_limit},{args.seed},{args.epsilon},{mean_err},{max_err}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset Parameters
    parser.add_argument('dataset', type=str, help='dataset name')
    # Privacy Parameters
    parser.add_argument('--epsilon', type=float, default=1.0, help='privacy parameter')
    parser.add_argument('--delta', type=float, default=1e-9, help='privacy parameter')
    # Workload Parameters
    parser.add_argument('--degree', type=int, default=3, choices=[1,2,3,4,5], help='degree of marginals in workload')
    # Rng Parameters
    parser.add_argument('--seed', type=int, default=17, help='prng seed')
    # Model Parameters
    parser.add_argument('--size_limit', type=float, default=80.0, help='model size limit')

    args = parser.parse_args()
    print(args, flush=True)
    main(args)