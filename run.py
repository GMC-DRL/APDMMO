from surrogate import *
from plot import *
from optimize import *
from np_parallel_cec2013.cec2013.cec2013 import *
import time
import torch
import os
from local_refine import *
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"


pid_list = range(1, 21)

for pid in pid_list:
    print('pid: {}'.format(pid))
    model_class = Surrogate_large
    time_stamp = time.strftime("%Y%m%dT%H%M%S")
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')
    if not os.path.exists('mu_std_info/'):
        os.makedirs('mu_std_info/')
    if not os.path.exists('log/'):
        os.makedirs('log/')
    log_dir = 'log/f' + str(pid)
    model_save_path = 'checkpoint/f'+str(pid)+'.pth'
    pic_dir = 'pic/f' + str(pid) + '/'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    device = 'cuda'
    problem = CEC2013(pid)

    sample_seed_training = 3
    model_seed = 333 
    learning_epochs = 400
    batch_size = 400
    namda = 3/8
    training_data_size = int(problem.get_maxfes() *namda)
    opt_surr_fit = torch.optim.AdamW
    lr_surr_fit = 0.0005

    # local optimization options
    lr_local_optimize = 0.005
    opt_local_optimization = torch.optim.AdamW
    sample_strategy = 'random'
    sample_seed_opt = 1
    sample_size = 1000000
    if problem.get_dimension() < 20:
        local_optimization_step = 3000
    else:
        local_optimization_step = 5000
    local_optimization_bc = 'reflect'  # reflect, clip
    obj_accuracy = 0.1

    

    # train surrogate
    print('pid: {}, true no goptima: {}'.format(pid, problem.get_no_goptima()))
    logger = SummaryWriter(log_dir)
    torch.manual_seed(model_seed)
    model = model_class(input_dim = problem.get_dimension(), hidden_dim=128, out_dim=1, sublayer_num=5).to(device)
    with open('model.txt', 'w') as f:
        print(model, file=f)

    model, DB_X, DB_Y, mu_x, std_x, mu_y, std_y = fit(problem,
                model,
                opt_surr_fit,
                lr_surr_fit,
                learning_epochs,
                sample_seed_training,
                device,
                training_data_size,
                batch_size,
                logger,
                model_save_path) 

    mu_std_info = {
        'mu_x': mu_x,
        'std_x': std_x,
        'mu_y': mu_y,
        'std_y': std_y
    }
    torch.save(mu_std_info, 'mu_std_info/f' + str(pid) + '.pth') # 'checkpoint/'+time_stamp+'.pth'
    if problem.get_dimension() == 1:
        plot_contour_origin_1D(problem,pic_dir, device)
        plot_contour_surrogate_1D(problem,model,pic_dir, device, mu_x, std_x, mu_y, std_y)
    if problem.get_dimension() == 2:
        plot_contour_origin(problem,pic_dir,device)
        plot_contour_surrogate(problem,model,pic_dir,device, mu_x, std_x, mu_y, std_y)
    

    # surrogate optimize phase
    if not os.path.exists('optimization_result/'):
        os.makedirs('optimization_result/')
    model = torch.load('checkpoint/f'+str(pid)+'.pth')
    mu_std_info = torch.load('mu_std_info/f' + str(pid) + '.pth')
    mu_x = mu_std_info['mu_x']
    std_x = mu_std_info['std_x']
    mu_y = mu_std_info['mu_y']
    std_y = mu_std_info['std_y']

    optimize_sampling = generate_optimize_sampling(sample_seed_opt,
                                                sample_size,
                                                sample_strategy,
                                                problem)
    optimas = []
    obj_optimas = []
    each_size = 100000
    for i in range(int(sample_size // each_size)):
        optima, obj_optima,_ = local_optimization_parallel(optimize_sampling[i*each_size: (i+1)*each_size], sample_seed_opt,
                                                sample_size,
                                                sample_strategy,
                                                local_optimization_step,
                                                device,model,
                                                lr_local_optimize,
                                                opt_local_optimization,
                                                local_optimization_bc,
                                                problem, mu_x, std_x, mu_y, std_y )

        optimas.append(optima.copy())
        obj_optimas.append(obj_optima.copy())
    optimas = np.concatenate(optimas, axis = 0)
    obj_optimas = np.concatenate(obj_optimas, axis = 0)
    assert len(optimas) == len(obj_optimas) == sample_size
    np.save('optimization_result/f'+str(pid) +'_opt_after_opt_'+ str(sample_size) +'.npy', optimas)
    np.save('optimization_result/f'+str(pid) +'_objopt_after_opt_'+ str(sample_size) +'.npy', obj_optimas)
    print('save complete!')

    # local search phase
    random_seed = 111
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if not os.path.exists('result/pr'):
        os.makedirs('result/pr')
    if not os.path.exists('result/sr'):
        os.makedirs('result/sr')
    optimas = np.load('optimization_result/f'+str(pid) +'_opt_after_opt_'+ str(sample_size) +'.npy')
    obj_optimas = np.load('optimization_result/f'+str(pid) +'_objopt_after_opt_'+ str(sample_size) +'.npy')
    assert optimas.shape == (sample_size, problem.get_dimension())

    seeds = find_seeds(problem, optimas, obj_optimas)

    print('start local search')

    run_time = 5
    peak_rate = np.zeros((20, run_time, 5))
    succ_rate = np.zeros((20, run_time,5))
    
    for i_run in tqdm(range(run_time), leave = True):
        ls_seed = np.random.randint(1, 1000)
        archive_pos = local_refine(problem, pid, seeds, max_fes=int(problem.get_maxfes() * (1-namda)), ls_seed = ls_seed)
        
        true_peak_num = problem.get_no_goptima()
        accuracy = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        for i_acc in range(5):
            count, _ = how_many_goptima(archive_pos, problem, accuracy[i_acc])
            peak_rate[pid - 1][i_run][i_acc] = count / true_peak_num
            if count >= true_peak_num:
                succ_rate[pid - 1][i_run][i_acc] = 1
        # print(peak_rate[pid-1][i_run])
    np.save('result/pr/f' + str(pid) +'.npy', peak_rate[pid-1])
    np.save('result/sr/f' + str(pid) +'.npy', succ_rate[pid-1])
    print(peak_rate[pid-1])
    print(succ_rate[pid-1])
    per_pr = np.mean(peak_rate[pid - 1], axis = 0)
    per_sr = np.mean(succ_rate[pid - 1], axis = 0)
    print(f'pid: {pid}, pr: {per_pr}, sr: {per_sr}')

