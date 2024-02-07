import torch
import os
import sys
import math
import numpy as np
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
 

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods



import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

#TODO: Curent hypothesis: not working because Cost has not full rank
#                         current solution is consider eps instead of 0
#TODO: setter function for variables
#TODO: x boundaries 
#TODO: x_init has to be in error dynamics, add the first state separately
#TODO: find out why last action is alwa7s [0, 0.1]:
#      Hypothesis: it has to do with x_init in pnqp. 
#TODO: check why doesn't match Ly des with actual state

class ALIPtorch_mpc():
    def __init__(self, robot, n_batch):     
        self.eps = 1e-4 
        self.n_state = 4
        self.n_ctrl = 2
        self.n_batch = n_batch
        self._robot = robot

        #TODO: set parameters from configuration file
        self.Ts = 0.25*torch.ones(self.n_batch)
        self.Tr = 0.25*torch.ones(self.n_batch)
        self.Nt = 4
        self.Ns = 4
        self.stance_sign = 1
        self.u_init = None
        self.mass = 39
        self._zH = torch.tensor(0.685)
        self.w = torch.tensor(0.1)
        self.g = torch.tensor(9.81)




        self.dt = self.Ts[0]/self.Nt

        #desired state
        self.Lx_offset = torch.zeros(self.n_batch)
        self.Ly_des = torch.zeros(self.n_batch)



        self.getLinDinF()

        self.getCost(True) #if True use eps instead of 0 

    
        #TODO: getter function from param data
        self.ufp_x_max = 0.6
        self.ufp_y_max = 0.4
        self.ufp_y_min = 0.05
        self.get_u_bounds()


    def solve_mpc_coor(self, stance_leg, x, Lx_offset, Ly_des, Tr): #x = [x, y, Lx, Ly]
        #computes x_0 as   A(Ts)_x
        self.Tr = Tr.clone()
        self.stance_sign = stance_leg.clone()
        self._x = x.clone()


        if (torch.allclose(Lx_offset, self.Lx_offset, atol = 1e-3)) or (torch.allclose(Ly_des, self.Ly_des, atol = 1e-3)):
            self.Lx_offset = Lx_offset.clone()
            self.Ly_des = Ly_des.clone()
            self.getCost(True)
        
        #computes x_0 //this is wrong
        #x_ = torch.cat((self._x, torch.zeros(n_batch,1), torch.zeros(n_batch,1)), dim = 1).unsqueeze(2)
        #x_0 = torch.bmm(self.F[0,:,:,:], x_).squeeze(2)

        exp_Atr = torch.linalg.matrix_exp(self._A.expand(self.n_batch,-1, -1)*self.Tr.view(-1, 1, 1)) 
        x = x.to(exp_Atr.dtype)
        x_0 = torch.matmul(exp_Atr, x.unsqueeze(2)).squeeze()
        

        #TODO: look at code status about the problem with this
        if(self.stance_sign[0] == 1): #set bounds TODO:CHECK f is the good one
            self.u_lower = self.u_lower_plus
            self.u_upper = self.u_upper_plus
            self.q = self.q_minus
        else: #first left swing  
            self.u_lower = self.u_lower_minus
            self.u_upper = self.u_upper_minus
            self.q = self.q_plus

        #self.u_lower = None
        #self.u_upper = None

        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
                self.n_state, self.n_ctrl, self.Ns,
                u_init= self.u_init,
                u_lower= self.u_lower, u_upper= self.u_upper,
                lqr_iter=50,
                verbose=-1,
                exit_unconverged=False,
                detach_unconverged=False,
                grad_method=GradMethods.ANALYTIC,
                eps=1e-2,
            )(x_0, QuadCost(self.Q, self.q), LinDx(self.F))
        self.up_u_init(nominal_actions)


        #change of coordinates
        return nominal_states, nominal_actions, nominal_objs
    
    def solve_inertia_coor(self, stance_leg, Lx_offset, Ly_des, Tr, torso_ori):
        pos = self._robot.get_com_pos()
        vel = self._robot.get_com_lin_vel()
        lfoot_pos = self._robot.get_link_iso("r_foot_contact")[:, 0:3, 3]
        rfoot_pos = self._robot.get_link_iso("l_foot_contact")[:, 0:3, 3]

        stleg_pos = torch.zeros(self.n_batch, 3)

        for i in range(self.n_batch):
            if(stance_leg[i] == 1):
                stleg_pos[i] = rfoot_pos[i]
            else:
                stleg_pos[i] = lfoot_pos[i]

        stleg_pos = stleg_pos.to(torso_ori.dtype)
        pos_torso_ori = torch.matmul(torso_ori.transpose(1,2), pos.unsqueeze(2)).squeeze()
        vel_torso_ori = torch.matmul(torso_ori.transpose(1,2), vel.unsqueeze(2)).squeeze()
        stleg_pos_torso_ori = torch.matmul(torso_ori.transpose(1,2), stleg_pos.unsqueeze(2)).squeeze()

        x = pos_torso_ori[:, 0:2] - stleg_pos_torso_ori[:, 0:2]
        #_x = torch.cat((x, self._zH), dim = 1) 

        L = self.mass*torch.linalg.cross(pos_torso_ori, vel_torso_ori)

        x = torch.cat((x, L[:, 0].unsqueeze(1), L[:, 1].unsqueeze(1)), dim = 1)
        states, actions, objc = self.solve_mpc_coor(stance_leg, x, Lx_offset, Ly_des, Tr)
        #For now assume height is constant
        next_action_torso_frame = torch.cat((actions[0, :, :], torch.zeros(self.n_batch, 1)), dim = 1)
        next_action_torso_frame = next_action_torso_frame.to(torso_ori.dtype)
        next_action = torch.matmul(torso_ori, next_action_torso_frame.unsqueeze(2)).squeeze() + stleg_pos

        return next_action

    #TODO: Control bounds don't seem to work properly
    def get_u_bounds(self): #-1 for LS solver
        assert self.Ns%2 == 0 #Ns need to be even in order to work with the current implementation of the u_bounds

        self.u_upper_plus = torch.tensor([[ self.ufp_x_max/2, self.ufp_y_max], 
                                     [ self.ufp_x_max/2, -self.ufp_y_min]])
        self.u_lower_plus = torch.tensor([[-self.ufp_x_max/2, self.ufp_y_min],
                                     [-self.ufp_x_max/2, -self.ufp_y_max]])

        self.u_upper_minus = torch.tensor([[self.ufp_x_max/2, -self.ufp_y_min], 
                                      [self.ufp_x_max/2, self.ufp_y_max]])
        self.u_lower_minus = torch.tensor([[-self.ufp_x_max/2,- self.ufp_y_max], 
                                      [-self.ufp_x_max/2, self.ufp_y_min]])

        """
        Since each robot can have different stance_legs, here they will not be batched
        """
        self.u_upper_plus = self.u_upper_plus.repeat(int(self.Ns/2), 1)
        self.u_upper_plus = self.u_upper_plus.unsqueeze(1).repeat(1,self.n_batch,1)

        self.u_lower_plus = self.u_lower_plus.repeat(int(self.Ns/2), 1)
        self.u_lower_plus = self.u_lower_plus.unsqueeze(1).repeat(1,self.n_batch,1)
    
        self.u_upper_minus = self.u_upper_minus.repeat(int(self.Ns/2), 1)
        self.u_upper_minus = self.u_upper_minus.unsqueeze(1).repeat(1,self.n_batch,1)

        self.u_lower_minus = self.u_lower_minus.repeat(int(self.Ns/2), 1)
        self.u_lower_minus = self.u_lower_minus.unsqueeze(1).repeat(1,self.n_batch,1)

        #self.u_lower_minus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)
        #self.u_lower_plus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)
        #self.u_upper_plus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)
        #self.u_upper_minus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)


    def up_u_init(self, u):
        """
        Erases first column and puts zeros at the back
        could try other approaches for the last column
        Problem: last column is not inside the bounds, if doesn't work try  u[0, :, :].unsqueeze(0) instead
        """
        #self.u_init = u #u[mpcT, nbatch, 2]
        self.u_init = torch.cat((u[1:self.Ns, :, :], torch.zeros(1, 3, 2)), dim = 0) 


    def getCost(self, bool_eps): #cost is checked 
        self.Qrunning = 2*torch.eye(self.n_state + self.n_ctrl)
        self.Qterminal = 2*100*torch.eye(self.n_state + self.n_ctrl)
        for i in range(self.n_state, self.n_state + self.n_ctrl): #i = 4, i =5
            if bool_eps:
                self.Qrunning[i,i] = self.eps
                self.Qterminal[i,i] = self.eps 
            else:
                self.Qrunning[i,i] = 0
                self.Qterminal[i,i] = 0     

        h = self.Qrunning.unsqueeze(0).unsqueeze(0).repeat(self.Ns-1, self.n_batch, 1, 1)
        Qt = self.Qterminal.repeat(1, self.n_batch, 1, 1)
        self.Q = torch.cat((h, Qt), 0)


        #desired state
        self.l = torch.sqrt(self.g/self._zH)
        q1 = -2/self.mass/self._zH/self.l * torch.tanh(self.l*self.Ts/2) * self.Ly_des
        #leg dependent desired state
        q2_plus = self.w
        q2_minus = -self.w
        q3_plus = -self.mass*self._zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) - 2*self.Lx_offset
        q3_minus = self.mass*self._zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) - 2*self.Lx_offset
        q4 = -2*self.Ly_des


        self.q_plus = torch.zeros(self.Ns, self.n_batch, self.n_state+self.n_ctrl)  #initial right stance /left swing
        self.q_minus = torch.zeros(self.Ns, self.n_batch, self.n_state+self.n_ctrl) #initial left stance /right swing

        self.q_plus[:,:,0] = self.q_minus[:,:,0] = q1
        self.q_plus[:,:,3] = self.q_minus[:,:,3] = q4

        for n in range(self.Ns-1):
            if n%2 == 0:
                self.q_plus[n, :, 1] = q2_plus
                self.q_minus[n, : ,1] = q2_minus
                self.q_plus[n, :, 2] = q3_plus
                self.q_minus[n, :,2] = q3_minus
            else:
                self.q_plus[n, :, 1] = q2_minus
                self.q_minus[n, : ,1] = q2_plus
                self.q_plus[n, :, 2] = q3_minus
                self.q_minus[n, : ,2] = q3_plus

        self.q_plus[self.Ns-1,:,0] = 100*q1
        self.q_minus[self.Ns-1,:,0] = 100*q1

        self.q_plus[self.Ns-1,:,1] = 100*self.q_minus[self.Ns-2,:,1]
        self.q_minus[self.Ns-1,:,1] = 100*self.q_plus[self.Ns-2,:,1]

        self.q_plus[self.Ns-1,:,2] = 100*self.q_minus[self.Ns-2,:,2]
        self.q_minus[self.Ns-1,:,2] = 100*self.q_plus[self.Ns-2,:,2]

        self.q_plus[self.Ns-1,:,3] = 100*q4
        self.q_minus[self.Ns-1,:,3] = 100*q4


    def getLinDinF(self):  #[xi+1] = self.F [xi, ui]  

        self._A = torch.tensor([[0 ,0,0, 1/self.mass/self._zH],
                          [0,0,-1/self.mass/self._zH,0],
                          [0,-self.mass*self.g,0,0],
                          [self.mass*self.g,0,0,0]])

        B = torch.tensor([[-1, 0],
                          [ 0,-1],
                          [ 0, 0],
                          [ 0, 0]], dtype=torch.float32)


        exp_At = torch.linalg.matrix_exp(self._A*self.dt) 
        AtB = torch.matmul(exp_At, B)

        self.F = torch.cat((exp_At, AtB), dim = 1)  
        self.F = self.F.repeat(self.Ns, self.n_batch, 1,1) 


    







    def get_frame(self, state):
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

    def plot_mpc_traj(self, state, u):
        fig = plt.subplots
        x, y, Lx, Ly = torch.unbind(state, dim = 2)
        ux, uy = torch.unbind(u, dim = 2)

        n_row = int(math.sqrt(self.n_batch))
        n_col = n_row+1

        #plot control evolution this is wrong
        ev_ux = torch.cumsum(ux, dim = 0)
        ev_uy = torch.cumsum(uy, dim = 0)
        print("ux", ux)
        print("ev_ux", ev_ux)

        ev_ux = torch.cat([torch.zeros(1, self.n_batch), ev_ux], dim=0)
        print(ev_ux)
        ev_uy = torch.cat([torch.zeros(1, self.n_batch), ev_uy], dim=0)
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






class indata():
    def __init__(self):
        self.Ts = 0.25
        self.Tr = 0.25
        self.mass = 39.15342
        self.zH = 0.7

        #self.state = torch.tensor([1,2,3,4])

        self.g = 9.81
        self.w = 0.13
        self.stance_leg = -1
        self.Lx_offset = 0.
        self.Ly_des = 6.

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low
    

if __name__ == '__main__':
    _indata = indata()
    n_batch, Ns = 6, 6   #Ns = T in mpc formulation

    n_state = 4
    n_ctrl = 2
    a = ALIP_mpc_torch_LinDx(Ns, 1, _indata, n_batch)
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











    

    

    






