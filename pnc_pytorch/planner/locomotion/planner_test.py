import torch
import math

import matplotlib as plt

from pnc_pytorch.planner.locomotion.alip_mpc import ALIPtorch_mpc
from config.draco3_alip_config import AlipParams

def get_frame(AlipMpc, state):
    assert len(state) == 6
    x, y, Lx, Ly , ufp, ufp2= torch.unbind(state)
    x = x.numpy()
    y = y.numpy()

    fig, ax = plt.subplots(figsize=(6,6))


    ax.scatter(x,y,color='k', marker = 'x', s=50)
    ax.set_xlim((-3, 3))
    ax.set_ylim((-3, 3))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')   
    return fig, ax

def plot_mpc_traj(state, u):
    n_batch = AlipParams.N_BATCH
    Ns = AlipParams.NS
    fig = plt.subplots
    x, y, Lx, Ly = torch.unbind(state, dim = 2)
    ux, uy = torch.unbind(u, dim = 2)

    n_row = int(math.sqrt(n_batch))
    n_col = n_row+1

    #plot control evolution this is wrong
    ev_ux = torch.cumsum(ux, dim = 0)
    ev_uy = torch.cumsum(uy, dim = 0)
    print("ux", ux)
    print("ev_ux", ev_ux)

    ev_ux = torch.cat([torch.zeros(1, n_batch), ev_ux], dim=0)
    print(ev_ux)
    ev_uy = torch.cat([torch.zeros(1, n_batch), ev_uy], dim=0)
    h_ev_ux_1 = ev_ux[0:self.Ns+1:2 , :]
    h_ev_ux_2 = ev_ux[1:self.Ns+1:2 , :]
    h_ev_uy_1 = ev_uy[0:self.Ns+1:2 , :]
    h_ev_uy_2 = ev_uy[1:self.Ns+1:2 , :]

    print("h 1", h_ev_ux_1)
    print("h 2", h_ev_ux_2)

    ev_x = x + ev_ux[:-1, :]
    ev_y = y + ev_uy[:-1, :]

    fig, ax = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
    for i in range(self.n_batch):
        row = i // n_col
        col = i% n_col
        ax[row,col].plot(h_ev_ux_1[:,i].numpy(), h_ev_uy_1[:,i].numpy(), marker = 'x', color = 'r', label = 'second swing leg')
        ax[row,col].plot(h_ev_ux_2[:,i].numpy(), h_ev_uy_2[:,i].numpy(), marker = 'x', color = 'b', label = 'first swing leg')
        ax[row,col].plot(ev_x[:,i].numpy(), ev_y[:,i].numpy(), marker = 'x', color = 'black', label = 'COM')
        ax[row,col].set_title(f'Batch {i}')
        ax[row, col].set_xlabel('X')
        ax[row, col].set_ylabel('Y')
        ax[row, col].set_aspect('equal')
        ax[row,col].legend()

    fig.suptitle(f'with starting leg {self.initial_stance_leg}') 
    plt.savefig('alip_mpc_pytorch_sol')

    fig, ax = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
    Lx_plus = 0.5*self.mass*self._zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) + self.Lx_offset
    Lx_minus = -0.5*self.mass*self._zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) + self.Lx_offset

    for i in range(self.n_batch):
        row = i // n_col
        col = i% n_col
        ax[row,col].plot(range(self.Ns), Lx[:,i].numpy(), marker = 'x', color = 'r', label = 'Lx')
        ax[row,col].plot(range(self.Ns), Ly[:,i].numpy(), marker = 'x', color = 'b', label = 'Ly')
        ax[row,col].plot(range(self.Ns), self.Ly_des*torch.ones(self.Ns).numpy() , color = 'cyan', label = 'Ly_des')
        ax[row,col].plot(range(self.Ns), Lx_plus*torch.ones(self.Ns).numpy(), color = 'orange', label = 'Lx_des')
        ax[row,col].plot(range(self.Ns), Lx_minus*torch.ones(self.Ns).numpy(), color = 'orange')
        ax[row,col].set_title(f'Batch {i}')
        ax[row, col].set_xlabel('X')
        ax[row, col].set_ylabel('Y')
        ax[row, col].set_aspect('equal')
        ax[row,col].legend()


    fig.suptitle(f'Ly_des = {self.Ly_des}; Lx = ')
    plt.savefig('alip_mpc_pytorch_angular')

    fig, ax = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
    for i in range(self.n_batch):
        row = i // n_col
        col = i% n_col
        ax[row,col].plot(range(self.Ns), ux[:,i].numpy(), marker = 'x', color = 'r', label = 'u_x')
        ax[row,col].plot(range(self.Ns), self.u_upper[:,i,0].numpy() , color = 'cyan', label = 'upper bound')
        ax[row,col].plot(range(self.Ns), self.u_lower[:,i,0].numpy(), color = 'orange', label = 'lower bound')
        ax[row,col].set_title(f'Batch {i}')
        ax[row, col].set_xlabel('X')
        ax[row, col].set_ylabel('Y')
        #ax[row, col].set_aspect('equal')
        ax[row,col].legend()


    fig.suptitle(f'Ux stance sign = {self.stance_sign}')
    plt.savefig('alip_mpc_control_x')
    """
    b1=self.u_upper_minus[:,i,1].numpy()
    b2=self.u_lower_minus[:,i,1].numpy()
    it=np.arange(-0.5,self.Ns-0.5,1)
    IT = np.c_[it[:-1], it[1:], it[1:]]
    B1 = np.c_[b1[:-1], b1[:-1], np.zeros_like(IT[:-1])*np.nan]
    B2 = np.c_[b2[:-1], b2[:-1], np.zeros_like(IT[:-1])*np.nan]
    """
    fig, ax = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
    for i in range(self.n_batch):
        a = np.arange(-0.5,self.Ns-0.5,1)
        row = i // n_col
        col = i% n_col
        ax[row,col].scatter(range(self.Ns), uy[:,i].numpy(), marker = 'x', color = 'r', label = 'u_y')
        ax[row,col].plot(np.arange(-0.5,self.Ns-0.5,1), self.u_upper[:,i,1].numpy(), drawstyle="steps-post", color = 'orange', label = 'upper bound')
        ax[row,col].plot(np.arange(-0.5,self.Ns-0.5,1), self.u_lower[:,i,1].numpy(), drawstyle="steps-post", color = 'orange', label = 'lower bound')
        ax[row,col].set_title(f'Batch {i}')
        ax[row, col].set_xlabel('X')
        ax[row, col].set_ylabel('Y')
        #ax[row, col].set_aspect('equal')
        ax[row,col].legend(loc='upper center')


    fig.suptitle(f'Uy stance sign = {self.stance_sign}')
    plt.savefig('alip_mpc_control_y')




def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low
    

if __name__ == '__main__':
    n_batch = AlipParams.N_BATCH

    alipMpc = ALIPtorch_mpc()


    solve_inertia_coor(stance_leg, Lx_offset, Ly_des, Tr, torso_ori):

    x = uniform(n_batch, -0.1, 0.1)
    #Lx = uniform(n_batch, -0.005, 0.005)
    #Ly = uniform(n_batch, -0.005, 0.005)
    Lx = torch.zeros(n_batch)
    Ly = torch.zeros(n_batch)
    
    torch.manual_seed(0)   
    if (_indata.stance_leg == -1): #left stance COMy < 0
        y = uniform(n_batch, -0.1, 0)

    else: #right stance COMy > 0
        y = uniform(n_batch, 0, 0.1)


    Lx = torch.zeros(n_batch)
    Ly = torch.zeros(n_batch)
    x = torch.zeros(n_batch)
    #y = torch.zeros(n_batch)

    x = torch.stack((x,y,Lx,Ly), dim = 1)


    nom_state, nom_u, nom_obj = a.solve(x, _indata)
    #print("state", nom_state)
    a.plot_mpc_traj(nom_state, nom_u)
    print(nom_u)

