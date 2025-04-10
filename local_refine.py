import numpy as np
import torch
from np_parallel_cec2013.cec2013.cec2013 import *
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import os, sys
from scipy.optimize import minimize
import copy
from pypop7.optimizers.es.sepcmaes import SEPCMAES


def find_seeds( problem, sols, obj_sols):
    dim = problem.get_dimension()
    assert len(sols) == len(obj_sols)
    # print(f'sols number after epsillon: {len(sols)}')

    lowbound = np.zeros(dim)
    upperbound = np.zeros(dim)
    for k in range(dim):
        upperbound[k] = problem.get_ubound(k)
        lowbound[k] = problem.get_lbound(k)

    if dim < 5:
        minsamples = 2
    elif dim == 5:
        minsamples = 20
    else:
        minsamples = 40
    
    if dim <= 5:
        eps = 0.1
    else:
        eps = 0.2
    print(f'eps: {eps}, minsaples: {minsamples}')

    seeds = []
    obj_seeds = []
    count = 0
    num_sub_batch = 100000
    for i_batch in range(int(len(sols) // num_sub_batch)): # batch for DBSCAN due to the limited memory
        sub_sols = sols[i_batch * num_sub_batch : (i_batch+1)* num_sub_batch].copy()
        sub_obj_sols = obj_sols[i_batch * num_sub_batch : (i_batch+1)* num_sub_batch].copy()

        clustering = DBSCAN(eps = eps, min_samples = minsamples).fit(sub_sols)
        cluster_labels = clustering.labels_

        count += len(sub_sols)
        for i in range(np.max(cluster_labels) +1):
            sub_cluster = np.where(cluster_labels == i)[0]
            min_index = np.argmin(sub_obj_sols[sub_cluster])
            seeds.append(sub_sols[sub_cluster][min_index].copy())
            obj_seeds.append(sub_obj_sols[sub_cluster][min_index])

    assert count == len(sols)
    seeds = np.array(seeds)
    obj_seeds = np.array(obj_seeds)
    assert len(seeds) == len(obj_seeds)

    eps_2 = eps
    remain_seeds = []
    remain_obj_seeds = []
    clustering = DBSCAN(eps = eps_2, min_samples = 1).fit(seeds)
    cluster_labels = clustering.labels_
    for i in range(np.max(cluster_labels) +1):
        sub_cluster = np.where(cluster_labels == i)[0]
        min_index = np.argmin(obj_seeds[sub_cluster])
        remain_seeds.append(seeds[sub_cluster][min_index].copy())
        remain_obj_seeds.append(obj_seeds[sub_cluster][min_index])
    seeds = np.array(remain_seeds)
    obj_seeds = np.array(remain_obj_seeds)

    print('<clustering> number of seeds: ', len(seeds))

    sorted_arg = np.argsort(obj_seeds)
    seeds = seeds[sorted_arg]
        
    return seeds

def tosolve_func(x, args):  # to define the fitness function to be minimized
    return -(args.evaluate(x[None,:])[0])


def local_refine(problem, pid, seeds, max_fes, ls_seed): # torch -> torch
    np.random.seed(ls_seed)
    torch.manual_seed(ls_seed)

    seed_len = len(seeds)
    dim = problem.get_dimension()
    lowbound = np.zeros(dim)
    upperbound = np.zeros(dim)
    for k in range(dim):
        upperbound[k] = problem.get_ubound(k)
        lowbound[k] = problem.get_lbound(k)
    gbest = problem.get_fitness_goptima()

    archive_accuracy = 0.1
    count = 0
    used_fes = 0
    
    if dim < 5:
        sigma = 0.1
    else:
        sigma = 0.5
    
    if dim < 5:
        popsize = 8
    elif dim < 20:
        popsize = 10
    else:
        popsize = 20

    max_function_evaluations = 200 * popsize
    
    early_stopping_evaluations = 20 * popsize

    # print(f'sigma: {sigma}, popsize:{popsize}, max_function_evaluations: {max_function_evaluations}, early_stop_fes: {early_stopping_evaluations}')
    archive_pos = []

    tosolve_problem = {'fitness_function': tosolve_func,  # define problem arguments
                        'ndim_problem': dim,
                        'lower_boundary': lowbound,
                        'upper_boundary': upperbound}
    
    seed_idx = 0
    while used_fes < max_fes:
        options = {'max_function_evaluations': max_function_evaluations,
                    'early_stopping_threshold': 1e-5,'early_stopping_evaluations': early_stopping_evaluations,
                    'mean': seeds[seed_idx],
                    'sigma': sigma,
                    'verbose': 0, 
                    'seed_rng': np.random.randint(1, 1000),
                    'n_individuals': popsize
                    }
                    
        cmaes = SEPCMAES(tosolve_problem, options)  # to initialize the optimizer class
        results = cmaes.optimize(args = problem)  # to run the optimization/evolution process

        archive_pos.append(results['best_so_far_x'])
        used_fes += results['n_function_evaluations']
        count += 1
        seed_idx = (seed_idx + 1) % len(seeds)

    archive_pos = np.array(archive_pos)
    # print('local refine use {} fes, max_fes: {}'.format(used_fes, max_fes))
    # print('local refine use {} points'.format(count))   

     
    return archive_pos
        
