import torch
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
# model = torch.load('checkpoint/20241007T181450.pth')

import math

def generate_optimize_sampling(seed,num_of_try,strategy,problem):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if strategy == 'grid':
        x = np.linspace(problem.get_lbound(0),problem.get_ubound(0),int(np.ceil(np.sqrt(num_of_try))))
        y = np.linspace(problem.get_lbound(0),problem.get_ubound(0),int(np.ceil(np.sqrt(num_of_try))))
        X,Y = np.meshgrid(x,y)
        temp = np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],-1)
    elif strategy == 'LHS':
        sampler = qmc.LatinHypercube(d=problem.get_dimension(),seed=seed)
        temp = problem.get_lbound(0) +(problem.get_ubound(0)- problem.get_lbound(0)) * sampler.random(num_of_try)
    elif strategy == 'random':
        temp = problem.get_lbound(0) + (problem.get_ubound(0) - problem.get_lbound(0)) * np.random.rand(num_of_try,problem.get_dimension())
    else:
        print('unknow strategy')
        assert False

    return temp

def local_optimization_parallel(temp, seed,num_of_try,strategy,num_of_step,device,model,lr,opt_class,bc,problem,mu_x, std_x, mu_y, std_y):
    print('Start Surrogate Optimization...')

    dim = problem.get_dimension()
    lowbound = torch.zeros(dim).to(device)
    upperbound = torch.zeros(dim).to(device)
    for k in range(dim):
        upperbound[k] = problem.get_ubound(k)
        lowbound[k] = problem.get_lbound(k)
    model.eval()


    x = torch.tensor(temp, dtype=torch.float32, device=device)
    x = (x - mu_x) / (std_x + 1e-20)
    x = x.detach().clone().requires_grad_(True)
    max_obj = torch.max(-(model(x.clone()).squeeze() * (std_y + 1e-20) + mu_y))
    upperbound = (upperbound - mu_x.squeeze()) / (std_x.squeeze() + 1e-20)
    lowbound = (lowbound - mu_x.squeeze()) / (std_x.squeeze() + 1e-20)
    opt = opt_class([x],lr=lr)
    pbar = tqdm(total=num_of_step)
    for epoch in range(num_of_step):
        objs = -model(x)
        loss = objs.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if bc == 'clip':
                # clip
                x[:] = x.clamp(problem.get_lbound(0),problem.get_ubound(0))
                    
            if bc == 'reflect':
                # reflect
                x[x>upperbound[None, :]] = (2 * upperbound[None,:] - x)[x>upperbound[None,:]]
                x[x<lowbound[None,:]] = (2 * lowbound[None, :] - x)[x<lowbound[None, :]]

        pbar.update()

    return (x * (std_x + 1e-20) + mu_x).cpu().data.numpy(), (-(model(x).squeeze() * (std_y + 1e-20) + mu_y)).cpu().data.numpy(), max_obj.cpu().data.numpy()