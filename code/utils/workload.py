import itertools
import json

def load_from_json(path):
    """ Load workload from json file """
    with open(path, 'r') as f:
        wl = json.load(f)
    return wl

def all_k_way(attrs, k):
    """ Return all k-way combinations of attributes """
    return list(itertools.combinations(attrs, k))

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(Ws):
    """ Compute the downward closure of a workload, i.e. the union of the power sets of all marginals """
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))