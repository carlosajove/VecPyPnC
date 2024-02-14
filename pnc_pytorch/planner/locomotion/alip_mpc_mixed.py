import torch
import os
import sys
import math
import numpy as np
 

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

from pnc_pytorch.data_saver import DataSaver
from config.draco3_alip_config import AlipParams

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

class ALIPmixedtorch_mpc():
    def __init__(self, robot, n_batch, data_save = False):     
        self.eps = 1e-4 
        self.n_state = 4
        self.n_ctrl = 2
        self.n_batch = n_batch
        self._robot = robot
        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()

        #TODO: set parameters from configuration file
        self.Ts = AlipParams.TS *torch.ones(self.n_batch)
        self.Tr = AlipParams.TS *torch.ones(self.n_batch)
        self.Nt = AlipParams.NT
        self.Ns = AlipParams.NS
        self.stance_sign = AlipParams.INITIAL_STANCE_LEG
        self.u_init = None
        self.mass = AlipParams.MASS
        self._zH = AlipParams.ZH
        self.w = AlipParams.WIDTH
        self.g = AlipParams.G




        self.dt = self.Ts[0]/self.Nt

        #desired state
        self.Lx_offset = AlipParams.LX_OFFSET * torch.ones(self.n_batch)
        self.Ly_des = AlipParams.LY_DES * torch.ones(self.n_batch)

        self._a = AlipParams.ALPHA
        self._b = 1 - self._a

        self.getLinDinF()

        self.getCost(True) #if True use eps instead of 0 

    
        #TODO: getter function from param data


        self.ufp_x_max = AlipParams.UFP_X_MAX
        self.ufp_y_max = AlipParams.UFP_Y_MAX
        self.ufp_y_min = AlipParams.UFP_Y_MIN

        self._ufp_x_max_com = AlipParams.UCOM_X_MAX
        self._ufp_y_ext = AlipParams.UCOM_Y_EXT
        self._ufp_y_int = AlipParams.UCOM_Y_INT
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
        #x_0 = x

        #TODO: look at code status about the problem with this
        if(self.stance_sign[0] == 1): #set bounds TODO:CHECK f is the good one
            self.u_lower = self.u_lower_plus
            self.u_upper = self.u_upper_plus
            self.q = self.q_minus
            
            #self.q = self.q_plus
        else: #first left swing  
            self.u_lower = self.u_lower_minus
            self.u_upper = self.u_upper_minus
            self.q = self.q_plus
            #self.q = self.q_minus

        self.u_lower = self.u_lower.unsqueeze(1).repeat(1, self.n_batch, 1)
        self.u_upper = self.u_upper.unsqueeze(1).repeat(1, self.n_batch, 1)

        #self.u_lower = None
        #self.u_upper = None

        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
                self.n_state, self.n_ctrl, self.Ns,
                u_init= self.u_init,
                u_lower= self.u_lower, u_upper= self.u_upper,
                lqr_iter=100,
                verbose=1,
                exit_unconverged=False,
                detach_unconverged=False,
                grad_method=GradMethods.ANALYTIC,
                eps=1e-2,
            )(x_0, QuadCost(self.Q, self.q), LinDx(self.F))
        self.up_u_init(nominal_actions)
        print("input", x_0)
        print("action: ", nominal_actions[0])

        if self._b_data_save:
            self._data_saver.add('mpc_actions', nominal_actions)
            self._data_saver.add('mpc_states', nominal_states)
            self._data_saver.add('mpc_x_0', x_0)  #initial state should be nominal_states[0]
            self._data_saver.add('mpc_Lx_offset', Lx_offset)
            self._data_saver.add('mpc_Ly_des', Ly_des)
            self._data_saver.add('mpc_stance_leg', stance_leg)
            self._data_saver.add('mpc_x', self._x)
            self._data_saver.add('mpc_Tr', Tr)
            self._data_saver.add('mpc_u_lower', self.u_lower)
            self._data_saver.add('mpc_u_upper', self.u_upper)
            self._data_saver.add('mpc_Cost', self.Q)
            self._data_saver.add('mpc_cost', self.q)
            self._data_saver.add('mpc_LinDx', self.F)


        #change of coordinates
        return nominal_states, nominal_actions, nominal_objs
    
    def solve_inertia_coor(self, stance_leg, Lx_offset, Ly_des, Tr, torso_ori,
                           pos, vel, lfoot_pos, rfoot_pos):
        stleg_pos = torch.zeros(self.n_batch, 3)
        for i in range(self.n_batch):
            if(stance_leg[i] == 1):
                stleg_pos[i] = rfoot_pos[i]
            else:
                stleg_pos[i] = lfoot_pos[i]

        #stleg_pos = stleg_pos.to(torso_ori.dtype)
        pos_torso_ori = torch.matmul(torso_ori.transpose(1,2), pos.unsqueeze(2)).squeeze()
        vel_torso_ori = torch.matmul(torso_ori.transpose(1,2), vel.unsqueeze(2)).squeeze()
        stleg_pos_torso_ori = torch.matmul(torso_ori.transpose(1,2), stleg_pos.unsqueeze(2)).squeeze()
        
        x = pos_torso_ori[:, 0:2] - stleg_pos_torso_ori[:, 0:2]

        if self._b_data_save:
            swfoot_pos = torch.zeros(self.n_batch, 3)
            for i in range(self.n_batch):
                if(stance_leg[i] == 1):
                    swfoot_pos[i] = lfoot_pos[i]
                else:
                    swfoot_pos[i] = rfoot_pos[i]
            swfoot_pos_ori = torch.matmul(torso_ori.transpose(1,2), swfoot_pos.unsqueeze(2)).squeeze()
            swf_pos = swfoot_pos_ori[:, 0:2] - stleg_pos_torso_ori[:, 0:2]
            self._data_saver.add('mpc_sw_foot_pos', swf_pos)


        _x = torch.cat((x, self._zH*torch.ones(self.n_batch).unsqueeze(1)), dim = 1) 
                
        vel_torso_ori[:,2] = torch.zeros(self.n_batch)
        print("position", _x)
        print("vel_torso_ori", vel_torso_ori)
        L = self.mass*torch.linalg.cross(_x, vel_torso_ori)

        x = torch.cat((x, L[:, 0].unsqueeze(1), L[:, 1].unsqueeze(1)), dim = 1)
        states, actions, objc = self.solve_mpc_coor(stance_leg, x, Lx_offset, Ly_des, Tr)
        #For now assume height is constant

        action = actions[0, :, :]
    
        next_action_torso_frame = action - self._a*x[:, 0:2]
        print("action", action)

        #next_action_torso_frame = actions[0, :, :] - x[:, 0:2]

        next_action_torso_frame = torch.cat((next_action_torso_frame, torch.zeros(self.n_batch, 1)), dim = 1)
        next_action_torso_frame = next_action_torso_frame.to(torso_ori.dtype)
        next_action = torch.matmul(torso_ori, next_action_torso_frame.unsqueeze(2)).squeeze() + stleg_pos








        next_action_torso_frame = torch.cat((actions[0, :, :], torch.zeros(self.n_batch, 1)), dim = 1)
        next_action_torso_frame = next_action_torso_frame.to(torso_ori.dtype)
        next_action = torch.matmul(torso_ori, next_action_torso_frame.unsqueeze(2)).squeeze() + stleg_pos

        return next_action


    def get_u_bounds(self): #-1 for left stance first --> starts with right swing
        u_upper_right_swing_com = torch.tensor([self._ufp_x_max_com/2, self._ufp_y_int]).unsqueeze(0)
        u_lower_right_swing_com = torch.tensor([-self._ufp_x_max_com/2, -self._ufp_y_ext]).unsqueeze(0)

        u_upper_left_swing_com = torch.tensor([self._ufp_x_max_com/2, self._ufp_y_ext]).unsqueeze(0)
        u_lower_left_swing_com = torch.tensor([-self._ufp_x_max_com/2, -self._ufp_y_int]).unsqueeze(0)


        u_upper_left_swing_st = torch.tensor([ self.ufp_x_max/2, self.ufp_y_max]) 
        u_lower_left_swing_st = torch.tensor([-self.ufp_x_max/2, self.ufp_y_min])

        u_upper_right_swing_st = torch.tensor([self.ufp_x_max/2, -self.ufp_y_min])
        u_lower_right_swing_st = torch.tensor([-self.ufp_x_max/2,- self.ufp_y_max])

        u_upper_right_swing = self._a*u_upper_right_swing_com + (self._b)*u_upper_right_swing_st
        u_lower_right_swing = self._a*u_lower_right_swing_com + (self._b)*u_lower_right_swing_st

        u_upper_left_swing = self._a*u_upper_left_swing_com + (self._b)*u_upper_left_swing_st
        u_lower_left_swing = self._a*u_lower_left_swing_com + (self._b)*u_lower_left_swing_st

        self.u_upper_plus = u_upper_left_swing
        self.u_lower_plus = u_lower_left_swing
        self.u_upper_minus = u_upper_right_swing
        self.u_lower_minus = u_lower_right_swing

        """
        Since each robot can have different stance_legs, here they will not be batched
        They will be batched afterwards
        """
        for i in range(self.Ns -1):
            if i%2 == 0:
                self.u_upper_plus = torch.cat((self.u_upper_plus, u_upper_right_swing), dim = 0)
                self.u_lower_plus = torch.cat((self.u_lower_plus, u_lower_right_swing), dim = 0)
                self.u_upper_minus = torch.cat((self.u_upper_minus, u_upper_left_swing), dim = 0)
                self.u_lower_minus = torch.cat((self.u_lower_minus, u_lower_left_swing), dim = 0)
            else:
                self.u_upper_plus = torch.cat((self.u_upper_plus, u_upper_left_swing), dim = 0)
                self.u_lower_plus = torch.cat((self.u_lower_plus, u_lower_left_swing), dim = 0)
                self.u_upper_minus = torch.cat((self.u_upper_minus, u_upper_right_swing), dim = 0)
                self.u_lower_minus = torch.cat((self.u_lower_minus, u_lower_right_swing), dim = 0)


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
        self.l = math.sqrt(self.g/self._zH)
        q1 = -2/self.mass/self._zH/self.l * torch.tanh(self.l*self.Ts/2) * self.Ly_des
        #leg dependent desired state
        q2_plus = 10*self.w
        q2_minus = -10*self.w
        q3_plus = -10*self.mass*self._zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) - 2*self.Lx_offset
        q3_minus = 10*self.mass*self._zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) - 2*self.Lx_offset
        q4 = -10*2*self.Ly_des


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

        self.q_plus[self.Ns-1,:,1] = 20*self.q_minus[self.Ns-2,:,1]
        self.q_minus[self.Ns-1,:,1] = 20*self.q_plus[self.Ns-2,:,1]

        self.q_plus[self.Ns-1,:,2] = 20*self.q_minus[self.Ns-2,:,2]
        self.q_minus[self.Ns-1,:,2] = 20*self.q_plus[self.Ns-2,:,2]

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

        IdminusalphaB = torch.tensor([[1+self._a, 0, 0, 0],
                                      [0, 1+self._a, 0, 0],
                                      [0,    0 , 1., 0],
                                      [0,    0, 0, 1.]])


        exp_At = torch.linalg.matrix_exp(self._A*self.dt)
        AtIdminusalphaB = torch.matmul(exp_At, IdminusalphaB) 

        AtB = torch.matmul(exp_At, B)

        self.F = torch.cat((AtIdminusalphaB, AtB), dim = 1)  
        self.F = self.F.repeat(self.Ns, self.n_batch, 1,1) 











    

    

    






