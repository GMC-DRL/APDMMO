import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import copy
from tqdm import tqdm

import math

class EmbeddingNet(nn.Module):
    
    def __init__(
            self,
            node_dim,
            embedding_dim,
        ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias = False)

    # def init_parameters(self):

    #     for param in self.parameters():
    #         stdv = 1. / math.sqrt(param.size(-1))
    #         param.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        h_em = self.embedder(x)
        return  h_em

class Sublayer(nn.Module):
    def __init__(self, hidden_dim):
        super(Sublayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.activations = nn.ModuleList([nn.ELU(),nn.Hardshrink(),nn.Hardsigmoid(), nn.Hardtanh(),nn.Hardswish(), \
                        nn.LeakyReLU(), nn.LogSigmoid(), nn.PReLU(),  nn.ReLU(), nn.ReLU6(), \
                        nn.RReLU(), nn.SELU(),nn.CELU(),  nn.GELU(),nn.Sigmoid(),  \
                        nn.SiLU(), nn.Mish(),nn.Softplus(), nn.Softshrink(), nn.Softsign(), \
                        nn.Tanh(), nn.Tanhshrink()])

        self.ln = nn.Linear(len(self.activations) * hidden_dim, hidden_dim)
        self.Norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x1 = [self.activations[i](x) for i in range(len(self.activations))]
        out = self.ln(torch.concatenate(x1, dim = 1))
        out = self.Norm(out + x)
        return out

class Surrogate_large(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, sublayer_num):
        super(Surrogate_large, self).__init__()

        self.embedding = EmbeddingNet(input_dim, hidden_dim)
        
        layers = []
        for i in range(sublayer_num):
            layers.append(Sublayer(hidden_dim=hidden_dim))
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        x = self.out(x)
        return x



def fit(problem, model, opt_class, lr, epoch,seed, device, ds, bs, logger, model_save_path): 
    print('Start Training...')
    model.train()
    torch.manual_seed(seed)
    np.random.seed(seed)
    nn_opt = opt_class(model.parameters(),lr=lr)

    DB_X = problem.get_lbound(0) + (problem.get_ubound(0) - problem.get_lbound(0)) * np.random.rand(ds,problem.get_dimension())
    DB_Y = problem.evaluate(DB_X)
    X_ = torch.from_numpy(DB_X).to(device).float()
    Y_ = torch.from_numpy(DB_Y).to(device).float()
    mu_x = torch.mean(X_, dim = 0, keepdim = True)
    std_x = torch.std(X_, dim = 0, keepdim = True)
    mu_y = torch.mean(Y_, dim = 0, keepdim = True)
    std_y = torch.std(Y_, dim = 0, keepdim = True)
    X_ = (X_ - mu_x) / (std_x + 1e-20)
    Y_ = (Y_ - mu_y) / (std_y + 1e-20)

    min_loss = 1e20
    pbar = tqdm(total=epoch)
    for epoch in range(epoch):
        shffule_index = torch.randperm(ds)
        X = X_[shffule_index]
        Y = Y_[shffule_index]

        layers = []
        count_each_layers = []
        min_Y = torch.min(Y).item()
        max_Y = torch.max(Y).item()
        num_layer = 10
        gap_Y = (max_Y - min_Y) / num_layer
        for i_layer in range(num_layer):
            layer_min = min_Y + i_layer * gap_Y
            layer_max = min_Y + (i_layer + 1) * gap_Y
            if i_layer == 0:
                layer_mask = ((Y >= layer_min) & (Y <= layer_max))
            else:
                layer_mask = ((Y > layer_min) & (Y <= layer_max))
            sub_idx = (torch.arange(ds).to(Y.device))[layer_mask]
            layers.append(sub_idx)
            count_each_layers.append(len(sub_idx))
            # print(layer_min, layer_max, len(sub_idx))
        assert np.sum(count_each_layers) == ds

        loss_sum = 0
        for bid in range(ds // bs):
            batch_choice = []
            for id_layer in range(len(layers)):
                num_choice = bs // num_layer
                start_index = (bid * num_choice) % len(layers[id_layer])
                choice_squeue = [(start_index + i) % len(layers[id_layer]) for i in range(num_choice)]
                a_choice = layers[id_layer][choice_squeue]
                batch_choice.append(a_choice)
            batch_choice = torch.cat(batch_choice)
            assert batch_choice.ndim == 1 and len(batch_choice) == bs

            x_train = X[batch_choice]
            y_pred = model(x_train)
            assert y_pred != None
            assert torch.isnan(y_pred).any() == False
            assert torch.isinf(y_pred).any() == False
            # print(y_pred.shape)
            y_targ = Y[batch_choice].unsqueeze(-1)
            # print(y_targ.shape)
            nn_opt.zero_grad()
            loss = F.mse_loss(y_pred, y_targ, reduction='mean')
            loss_sum += loss.detach().cpu().item()
            # print(loss.shape)
            if logger is not None:
                logger.add_scalar('loss', loss.cpu().item(), epoch * (ds//bs) + bid)
            loss.backward()
            nn_opt.step()
        if (loss_sum / (ds // bs))  < min_loss:
            min_loss = (loss_sum / (ds // bs))
            torch.save(model,model_save_path)
            best_model = copy.deepcopy(model)
        pbar.update()
    return best_model, DB_X, DB_Y, mu_x, std_x, mu_y, std_y 




    
    

            



