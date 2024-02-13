import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import torch
import math

import matplotlib
import matplotlib.pyplot as plt

from config.draco3_alip_config import AlipParams

folder_path = 'plot/alip_mpc_plots/'

def plot_mpc_traj(idx, state, u, n_batch, Ns, stance_sign,
                  Lx_offset, Ly_des, u_lower, u_upper, 
                  robot_x = None, initial_sw_foot_pos = None):

    mass = AlipParams.MASS
    n_batch = AlipParams.N_BATCH
    Ts = AlipParams.TS 
    zH = AlipParams.ZH
    g = AlipParams.G
    w = AlipParams.WIDTH
    l = math.sqrt(g/zH)



    fig = plt.subplots
    x, y, Lx, Ly = torch.unbind(state, dim = 2)
    ux, uy = torch.unbind(u, dim = 2)

    n_row = math.ceil(math.sqrt(n_batch))
    n_col = n_row

    #plot control evolution this is wrong
    ev_ux = torch.cumsum(ux, dim = 0)
    ev_uy = torch.cumsum(uy, dim = 0)

    
    ev_ux = torch.cat([torch.zeros(1, n_batch), ev_ux], dim=0)
    ev_uy = torch.cat([torch.zeros(1, n_batch), ev_uy], dim=0)

    h_ev_ux_1 = ev_ux[0:Ns+1:2 , :]
    h_ev_ux_2 = ev_ux[1:Ns+1:2 , :]
    h_ev_uy_1 = ev_uy[0:Ns+1:2 , :]
    h_ev_uy_2 = ev_uy[1:Ns+1:2 , :]

    if initial_sw_foot_pos is not None:
        h_ev_ux_2 = torch.cat((initial_sw_foot_pos[:, 0].unsqueeze(0), h_ev_ux_2), dim = 0)
        h_ev_uy_2 = torch.cat((initial_sw_foot_pos[:, 1].unsqueeze(0), h_ev_uy_2), dim = 0)
    
    ev_x = x + ev_ux[:-1, :]
    ev_y = y + ev_uy[:-1, :]
    fig1, ax1 = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
    for i in range(n_batch):
        row = i // n_col
        col = i% n_col
        h = h_ev_ux_1[:,0].numpy()
        y = h_ev_uy_1[:,0].numpy()

        ax1[row, col].plot(h_ev_ux_1[:, i], h_ev_uy_1[:, i], marker = 'x', color = 'r', label = 'second swing leg')
        ax1[row, col].plot(h_ev_ux_2[:, i].numpy(), h_ev_uy_2[:,i].numpy(), marker = 'x', color = 'b', label = 'first swing leg')
        ax1[row, col].plot(ev_x[:,i].numpy(), ev_y[:,i].numpy(), marker = 'x', color = 'black', label = 'COM')

        ax1[row, col].scatter(h_ev_ux_1[0, i], h_ev_uy_1[0, i], marker = '8', color = 'r')
        ax1[row, col].scatter(h_ev_ux_2[0, i].numpy(), h_ev_uy_2[0,i].numpy(), marker = 'o', color = 'b')
        ax1[row, col].scatter(ev_x[0,i].numpy(), ev_y[0,i].numpy(), marker = 'v', color = 'black')
        if robot_x is not None:
            ax1[row, col].scatter(robot_x[i, 0], robot_x[i, 1], marker = 'D', color = 'y')
        
        ax1[row, col].set_title(f'Batch {i}')
        ax1[row, col].set_xlabel('X')
        ax1[row, col].set_ylabel('Y')
        #ax1[row, col].set_aspect('equal', adjustable = 'box')
        ax1[row, col].legend()

    fig1.suptitle(f'with starting leg {stance_sign}') 
    plt.savefig(f'{folder_path}alip_mpc_pytorch_sol{idx:03}')

    fig2, ax2 = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
    Lx_plus = 0.5*mass*zH*l*w*math.sqrt(l*Ts*0.5) + Lx_offset
    Lx_minus = -0.5*mass*zH*l*w*math.sqrt(l*Ts*0.5) + Lx_offset

    for i in range(n_batch):
        row = i // n_col
        col = i% n_col
        ax2[row,col].plot(range(Ns), Lx[:,i].numpy(), marker = 'x', color = 'r', label = 'Lx')
        ax2[row,col].plot(range(Ns), Ly[:,i].numpy(), marker = 'x', color = 'b', label = 'Ly')
        ax2[row,col].plot(range(Ns), Ly_des[i]*torch.ones(Ns).numpy() , color = 'cyan', label = 'Ly_des')
        ax2[row,col].plot(range(Ns), Lx_plus[i]*torch.ones(Ns).numpy(), color = 'orange', label = 'Lx_des')
        ax2[row,col].plot(range(Ns), Lx_minus[i]*torch.ones(Ns).numpy(), color = 'orange')
        if stance_sign[i] == 1: #Right stance
            n = range(0, Ns, 2)
            ax2[row,col].scatter(n, Lx_minus[i]*torch.ones(len(n)).numpy(), marker = 'x', color = 'orange')
            n = range(1, Ns, 2)
            ax2[row,col].scatter(n, Lx_plus[i]*torch.ones(len(n)).numpy(), marker = 'x', color = 'orange')
        else:
            n = range(0, Ns, 2)
            ax2[row,col].scatter(n, Lx_plus[i]*torch.ones(len(n)).numpy(), marker = 'x', color = 'orange')
            n = range(1, Ns, 2)
            ax2[row,col].scatter(n, Lx_minus[i]*torch.ones(len(n)).numpy(), marker = 'x', color = 'orange') 

        ax2[row,col].set_title(f'Batch {i}')
        ax2[row, col].set_xlabel('X')
        ax2[row, col].set_ylabel('Y')
        #ax2[row, col].set_aspect('equal')
        ax2[row,col].legend()
    fig2.suptitle(f'Ly_des = {Ly_des}; Lx = ')
    plt.savefig(f'{folder_path}alip_mpc_pytorch_angular{idx:03}')


    if u_lower is not None:
        fig3, ax3 = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
        for i in range(n_batch):
            row = i // n_col
            col = i% n_col
            ax3[row,col].plot(range(Ns), ux[:,i].numpy(), marker = 'x', color = 'r', label = 'u_x')
            ax3[row,col].plot(range(Ns), u_upper[:,i,0].numpy() , color = 'cyan', label = 'upper bound')
            ax3[row,col].plot(range(Ns), u_lower[:,i,0].numpy(), color = 'orange', label = 'lower bound')
            ax3[row,col].set_title(f'Batch {i}')
            ax3[row, col].set_xlabel('X')
            ax3[row, col].set_ylabel('Y')
            #ax[row, col].set_aspect('equal')
            ax3[row,col].legend()


        fig3.suptitle(f'Ux stance sign = {stance_sign}')
        plt.savefig(f'{folder_path}alip_mpc_control_x{idx:03}')
    """
    b1=u_upper_minus[:,i,1].numpy()
    b2=u_lower_minus[:,i,1].numpy()
    it=np.arange(-0.5,Ns-0.5,1)
    IT = np.c_[it[:-1], it[1:], it[1:]]
    B1 = np.c_[b1[:-1], b1[:-1], np.zeros_like(IT[:-1])*np.nan]
    B2 = np.c_[b2[:-1], b2[:-1], np.zeros_like(IT[:-1])*np.nan]
    """
    if u_lower is not None:
        fig4, ax4 = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
        for i in range(n_batch):
            row = i // n_col
            col = i% n_col
            ax4[row,col].scatter(range(Ns), uy[:,i].numpy(), marker = 'x', color = 'r', label = 'u_y')
            ax4[row,col].plot(np.arange(-0.5,Ns-0.5,1), u_upper[:,i,1].numpy(), drawstyle="steps-post", color = 'orange', label = 'upper bound')
            ax4[row,col].plot(np.arange(-0.5,Ns-0.5,1), u_lower[:,i,1].numpy(), drawstyle="steps-post", color = 'orange', label = 'lower bound')
            ax4[row,col].set_title(f'Batch {i}')
            ax4[row, col].set_xlabel('X')
            ax4[row, col].set_ylabel('Y')
            #ax[row, col].set_aspect('equal')
            ax4[row,col].legend(loc='upper center')


        fig4.suptitle(f'Uy stance sign = {stance_sign}')
        plt.savefig(f'{folder_path}alip_mpc_control_y{idx:03}')

    fig5, ax5 = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
    for i in range(n_batch):
        ax5[row,col].plot(u[:, i, 0], u[:, i, 1])
        ax5[row,col].set_title(f'actions {i}')

    fig5.suptitle(f'Uy stance sign = {stance_sign}')
    plt.savefig(f'{folder_path}actions{idx:03}')







if __name__ == '__main__':
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    n_batch = AlipParams.N_BATCH
    Ns = AlipParams.NS
    actions = []
    states = []
    mpc_initial_params = ["Lx_offset", "Ly_des", "stance_leg", "x_0", "x", "Tr",
                        "u_lower", "u_upper", "Cost", "cost", "LinDx", "sw_foot_pos"]

    initial_params = dict()
    for param in mpc_initial_params:
        initial_params[param] = []
    

    with open('data/pnc.pkl', 'rb') as file:
        while True:
            try:
                d = pickle.load(file)
                if 'mpc_actions' in d:
                    actions.append(d['mpc_actions'])
                    states.append(d['mpc_states'])
                    for param in mpc_initial_params:
                        initial_params[param].append(d['mpc_' + param])
            except EOFError:
                break
        
    
    for idx in range(len(actions)):
        plot_mpc_traj(idx,states[idx], actions[idx], n_batch, Ns, initial_params['stance_leg'][idx], initial_params['Lx_offset'][idx],
                        initial_params['Ly_des'][idx], initial_params['u_lower'][idx], initial_params['u_upper'][idx],
                        initial_params['x'][idx], initial_params['sw_foot_pos'][idx])

