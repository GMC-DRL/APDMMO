import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import torch


def plot_contour_origin(problem,log_path,device):
    plt.figure()
    x = np.linspace(problem.get_lbound(0),problem.get_ubound(0),256)
    y = np.linspace(problem.get_lbound(0),problem.get_ubound(0),256)
    X,Y = np.meshgrid(x,y)
    temp = np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],-1)
    Z = -problem.evaluate(temp).reshape(256,256)
    plt.contour(X,Y,Z)
    plt.savefig(log_path+"original_contour.png")

    plt.close()

def plot_contour_surrogate(problem,model,log_path,device, mu_x, std_x, mu_y, std_y):
    plt.figure()
    model.eval()
    # model = torch.load('checkpoint/20241007T181450.pth')
    x = np.linspace(problem.get_lbound(0),problem.get_ubound(0),256)
    y = np.linspace(problem.get_lbound(0),problem.get_ubound(0),256)
    X,Y = np.meshgrid(x,y)
    temp = torch.tensor(np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],-1), dtype=torch.float32).to(device)
    temp = (temp - mu_x) / (std_x + 1e-20)
    Z = -(model(temp).squeeze() * (std_y + 1e-20) + mu_y) 
    Z = Z.detach().cpu().numpy().reshape(256,256)
    plt.contour(X,Y,Z)
    plt.savefig(log_path+'surrogate_contour.png')

    plt.close()


def plot_contour_origin_1D(problem,log_path, device):
    plt.figure()
    X=np.linspace(problem.get_lbound(0),problem.get_ubound(0),256)  
    Y = -problem.evaluate(X[:, None])      

    
    plt.plot(X,Y)       

    plt.savefig(log_path+"original_contour.png")

    plt.close()

def plot_contour_surrogate_1D(problem,model,log_path, device, mu_x, std_x, mu_y, std_y):
    plt.figure()
    model.eval()
    # model = torch.load('checkpoint/20241007T181450.pth')
    X=np.linspace(problem.get_lbound(0),problem.get_ubound(0),256) 
    temp = torch.tensor(X, dtype=torch.float32).to(device)
    temp = (temp[:,None] - mu_x) / (std_x + 1e-20)
    Y = -(model(temp).squeeze() * (std_y + 1e-20) + mu_y) 
    Y = Y.detach().cpu().numpy()
    plt.plot(X,Y)  
    plt.savefig(log_path+'surrogate_contour.png')

    plt.close()

