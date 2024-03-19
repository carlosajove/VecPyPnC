import torch
import os
import sys
import math
import numpy as np


from pnc_pytorch.data_saver import DataSaver
from qpth.qp import QPFunction   #for now like this for testing 
from qpsolvers import solve_qp


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

class ALIPtorch_mpc():
    def __init__(self, robot = None, data_save = False):     
        self.eps = 1e-8
        self.n_state = 4
        self.n_ctrl = 2
        self._robot = robot
        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()
        #TODO: set parameters from configuration file
        self._Ts = AlipParams.TS 
        self._Tr = AlipParams.TS 
        self._Nt = AlipParams.NT_qp
        self._Ns = AlipParams.NS
        self._mass = AlipParams.MASS
        self._zH = AlipParams.ZH
        self._w = AlipParams.WIDTH
        self._g = AlipParams.G
        self._dt = self._Ts/self._Nt

        #for now don't take too much into acount slip limit, as we are considering flat terrain
        self._kx = 0.
        self._ky = 0.
        self._mu = 0.4
        self._xc_slip_limit = (self._mu - self._kx)*self._zH / (1. + self._kx*self._kx);
        self._yc_slip_limit = (self._mu - self._ky)*self._zH / (1. + self._ky*self._ky);



        #number of optvariables: (Ns*Nt)*4 + Ns*2
        self.getLinDinF()
        self.equality_constraint_mat()

        #Get PARAMS FOR COST
        self._px = AlipParams.PX
        self._py = AlipParams.PY
        self._pLx = AlipParams.PLX
        self._pLy = AlipParams.PLY

        self.cost_mat()
    
        #TODO: getter function from param data
        self._ufp_x_max = AlipParams.UFP_X_MAX
        self._ufp_y_max = AlipParams.UFP_Y_MAX
        self._ufp_y_min_norm = AlipParams.UFP_Y_MIN
        self._ufp_y_min_turning = AlipParams.UFP_Y_MIN_turn
        self._x_mech_limit = self._ufp_x_max/2
        #self.static_u_bounds()
        self.ineq_mat()


      

    def solve_mpc_coor(self, stance_leg, x, Lx_offset, Ly_des, Tr): #x = [x, y, Lx, Ly]
        #computes x_0 as   A(Ts)_x
        self._batch = x.shape[0]
        self.Tr = Tr.clone()
        self.static_u_bounds()
        #DYNAMICS

        #STANCE LEG IDX
        mask = torch.eq(stance_leg, 1)
        idx_plus = torch.nonzero(mask).squeeze().tolist()
        idx_minus = torch.nonzero(~mask).squeeze().tolist()
        idx_plus = [idx_plus] if not isinstance(idx_plus, list) else idx_plus
        idx_minus = [idx_minus] if not isinstance(idx_minus, list) else idx_minus

        #COST AND BOUNDS
        self.Lx_offset = Lx_offset.clone()
        self.Ly_des = Ly_des.clone()

        self.equality_contraint_vec(x, self.Tr)
        self.cost_vec(idx_plus, idx_minus)
        self.ineq_vec(idx_plus, idx_minus)

        #self.u_lower = None
        #self.u_upper = None
        
        sol = QPFunction(verbose = -1)(self._cost_mat, self._cost_vec, 
                                       self._ineq_cont_mat, self._ineq_cont_vec, 
                                       self._eq_cont_mat, self._eq_cont_vec)

        state_sol = sol[:, 0:self._Ns*self._Nt*self.n_state]
        state_sol = state_sol.reshape(self._batch, state_sol.size(1)//self.n_state, self.n_state)
        state_sol = state_sol.transpose(0,1)
        u_sol = sol[:, self._Ns*self._Nt*self.n_state:self._Ns*self._Nt*self.n_state+self._Ns*self.n_ctrl]
        u_sol = u_sol.reshape(self._batch, u_sol.size(1)//self.n_ctrl, self.n_ctrl)
        u_sol = u_sol.transpose(0, 1)

        #clamp the solutions
        actions = torch.clamp(u_sol, self.u_lower, self.u_upper)

        if self._b_data_save:
            
            self._data_saver.add('mpc_actions', actions)
            self._data_saver.add('mpc_u_sol', u_sol)
            self._data_saver.add('mpc_states', state_sol)
            self._data_saver.add('mpc_x_0', x)  #initial state should be nominal_states[0]
            self._data_saver.add('mpc_Lx_offset', Lx_offset)
            self._data_saver.add('mpc_Ly_des', Ly_des)
            self._data_saver.add('mpc_stance_leg', stance_leg)
            self._data_saver.add('mpc_x', x.clone())
            self._data_saver.add('mpc_Tr', Tr)
            self._data_saver.add('mpc_u_lower', self.u_lower)
            self._data_saver.add('mpc_u_upper', self.u_upper)
            self._data_saver.add('mpc_cost_mat', self._cost_mat)
            self._data_saver.add('mpc_cost_vec', self._cost_mat)
            self._data_saver.add('mpc_ineq_mat', self._ineq_cont_mat)
            self._data_saver.add('mpc_ineq_vec', self._ineq_cont_vec)
            self._data_saver.add('mpc_eq_mat', self._ineq_cont_mat)
            self._data_saver.add('mpc_eq_vec', self._ineq_cont_vec)

        return state_sol, actions



    
    def solve_inertia_coor(self, stance_leg, Lx_offset, Ly_des, Tr, torso_ori,
                           pos, vel, lfoot_pos, rfoot_pos, turn_ids):
        self._batch = pos.shape[0]

        stleg_pos = torch.where(stance_leg.unsqueeze(1) == 1, rfoot_pos, lfoot_pos)

        #stleg_pos = stleg_pos.to(torso_ori.dtype)
        pos_torso_ori = torch.matmul(torso_ori.transpose(1,2), pos.unsqueeze(2)).squeeze()
        vel_torso_ori = torch.matmul(torso_ori.transpose(1,2), vel.unsqueeze(2)).squeeze()
        stleg_pos_torso_ori = torch.matmul(torso_ori.transpose(1,2), stleg_pos.unsqueeze(2)).squeeze()
        
        x = pos_torso_ori[:, 0:2] - stleg_pos_torso_ori[:, 0:2]

        """
        if self._b_data_save:
            swfoot_pos = torch.where(stance_leg.unsqueeze(1) == 1, lfoot_pos, rfoot_pos)
            swfoot_pos_ori = torch.matmul(torso_ori.transpose(1,2), swfoot_pos.unsqueeze(2)).squeeze()
            swf_pos = swfoot_pos_ori[:, 0:2] - stleg_pos_torso_ori[:, 0:2]
            self._data_saver.add('mpc_sw_foot_pos', swf_pos)
        """

        _x = torch.cat((x, self._zH*torch.ones(self._batch).unsqueeze(1)), dim = 1) 
                
        vel_torso_ori[:,2] = torch.zeros(self._batch, dtype = torch.double)

        L = self._mass*torch.linalg.cross(_x, vel_torso_ori)

        x = torch.cat((x, L[:, 0].unsqueeze(1), L[:, 1].unsqueeze(1)), dim = 1)


        states, actions = self.solve_mpc_coor(stance_leg, x, Lx_offset, Ly_des, Tr)
        #For now assume height is constant
        next_action_torso_frame = torch.cat((actions[0, :, :], torch.zeros(self._batch, 1, dtype = torch.double)), dim = 1)
        next_action_torso_frame = next_action_torso_frame.to(torso_ori.dtype)
        next_action = torch.matmul(torso_ori, next_action_torso_frame.unsqueeze(2)).squeeze() + stleg_pos

        return next_action

    def static_u_bounds(self): #-1 for left stance first --> starts with right swing
        assert self._Ns%2 == 0 #Ns need to be even in order to work with the current implementation of the u_bounds

        u_upper_left_swing = torch.tensor([self._ufp_x_max/2, self._ufp_y_max], dtype = torch.double).unsqueeze(0)
        u_lower_left_swing = torch.tensor([-self._ufp_x_max/2, self._ufp_y_min_norm], dtype = torch.double).unsqueeze(0)

        u_upper_right_swing = torch.tensor([self._ufp_x_max/2, -self._ufp_y_min_norm], dtype = torch.double).unsqueeze(0)
        u_lower_right_swing = torch.tensor([-self._ufp_x_max/2, -self._ufp_y_max], dtype = torch.double).unsqueeze(0)

        self.u_upper_plus = u_upper_left_swing
        self.u_lower_plus = u_lower_left_swing
        self.u_upper_minus = u_upper_right_swing
        self.u_lower_minus = u_lower_right_swing


        """
        Since each robot can have different stance_legs, here they will not be batched
        They will be batched afterwards
        """

        for i in range(self._Ns -1):
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


    def get_u_bounds(self, idp, idm):
        self.u_upper = torch.zeros(self._Ns, self._batch, self.n_ctrl, dtype = torch.double)
        self.u_lower = torch.zeros(self._Ns, self._batch, self.n_ctrl, dtype = torch.double)

        if idp:
            self.u_upper[:, idp] = self.u_upper_plus.unsqueeze(1).repeat(1, len(idp), 1)
            self.u_lower[:, idp] = self.u_lower_plus.unsqueeze(1).repeat(1, len(idp), 1)

        if idm:
            self.u_upper[:, idm] = self.u_upper_minus.unsqueeze(1).repeat(1, len(idm), 1)
            self.u_lower[:, idm] = self.u_lower_minus.unsqueeze(1).repeat(1, len(idm), 1)

    def up_u_init(self, u):
        """
        Erases first column and puts zeros at the back
        could try other approaches for the last column
        Problem: last column is not inside the bounds, if doesn't work try  u[0, :, :].unsqueeze(0) instead
        """
        #self.u_init = u #u[mpcT, nbatch, 2]
        self.u_init = torch.cat((u[1:self._Ns, :, :], torch.zeros(1, self._batch, 2, dtype = torch.double)), dim = 0) 


    def getCostvec(self, idp, idm): #cost is checked 
        #desired state
        self.l = math.sqrt(self._g/self._zH)

        print("width", self._w)
        q1 = self._px*(-2/self._mass/self._zH/self.l * math.tanh(self.l*self._Ts/2) * self.Ly_des)
        #leg dependent desired state
        q2_plus = self._py * self._w * torch.ones(self._batch, dtype = torch.double)
        q2_minus = self._py * (-self._w) * torch.ones(self._batch, dtype = torch.double)
        q3_plus = self._pLx * (-self._mass*self._zH*self.l*self._w*math.sqrt(self.l*self._Ts*0.5) - 2*self.Lx_offset)
        q3_minus = self._pLx * (self._mass*self._zH*self.l*self._w*math.sqrt(self.l*self._Ts*0.5) - 2*self.Lx_offset)
        q4 = self._pLy * (-2*self.Ly_des)

        self.q = torch.zeros(self._Ns, self._batch, self.n_state+self.n_ctrl, dtype = torch.double)  #initial right stance /left swing

        self.q[:,:,0] = q1.repeat(self._Ns, 1)
        self.q[:,:,3] = q4.repeat(self._Ns, 1)

        #idp st_leg + -- > q_plus
        #idm st_leg - -- > q_minus
        a = 1
        if (len(idp) == 0):
            for n in range(self._Ns):
                if (n == self._Ns-1):
                    a = 100
                if n%2 == 0:
                    self.q[n, :, 1] = a*q2_minus
                    self.q[n, :, 2] = a*q3_minus
                else:
                    self.q[n, :,1] = a*q2_plus
                    self.q[n, :,2] = a*q3_plus
        elif (len(idm) == 0):
            for n in range(self._Ns):
                if (n == self._Ns-1):
                    a = 100
                if n%2 == 0:
                    self.q[n, :, 1] = a*q2_plus
                    self.q[n, :, 2] = a*q3_plus
                else:
                    self.q[n, :, 1] = a*q2_minus
                    self.q[n, :, 2] = a*q3_minus
        else:
            for n in range(self._Ns):
                if (n == self._Ns-1):
                    a = 100
                if n%2 == 0:
                    self.q[n, idp, 1] = a*q2_plus[idp]
                    self.q[n, idm, 1] = a*q2_minus[idm]
                    self.q[n, idp, 2] = a*q3_plus[idp]
                    self.q[n, idm, 2] = a*q3_minus[idm]
                else:
                    self.q[n, idp, 1] = a*q2_minus[idp]
                    self.q[n, idm,1] = a*q2_plus[idm]
                    self.q[n, idp, 2] = a*q3_minus[idp]
                    self.q[n, idm,2] = a*q3_plus[idm]

        self.q[self._Ns-1,:,0] = 100*q1
        self.q[self._Ns-1,:,3] = 100*q4





    def getLinDinF(self):  #[xi+1] = self.F [xi, ui]  

        self._A = torch.tensor([[0 ,0,0, 1/self._mass/self._zH],
                          [0,0,-1/self._mass/self._zH,0],
                          [0,-self._mass*self._g,0,0],
                          [self._mass*self._g,0,0,0]], dtype = torch.double)

        self._B = torch.tensor([[-1, 0],
                          [ 0,-1],
                          [ 0, 0],
                          [ 0, 0]], dtype=torch.double)


        self._exp_At = torch.linalg.matrix_exp(self._A*self._dt) 
        self._exp_At_B = torch.matmul(self._exp_At, self._B)


    def equality_constraint_mat(self): #not batched
        #size of mat: 
        # vertical: (Ns*Nt)*4 -->>n_eq
        # horizontal: (Ns*Nt)*4 + Ns*2 -->> optimizatin variables
        state_mat = torch.cat((torch.eye(4, dtype = torch.double), torch.zeros(4, (self._Ns*self._Nt - 1)*4, dtype = torch.double)), dim = 1)
        mat = torch.cat((-self._exp_At, torch.eye(4, dtype = torch.double)), dim = 1)
        state_mat = torch.cat((state_mat, torch.cat((mat, torch.zeros(4, self._Ns*self._Nt*self.n_state - 2*self.n_state, dtype = torch.double)), dim = 1)), dim = 0)
        left, right = self.n_state, self._Ns*self._Nt*4 - self.n_state*3

        for i in range(self._Ns*self._Nt - 2):
            state_mat = torch.cat((state_mat, 
                                   torch.cat((torch.zeros(4, left, dtype = torch.double),
                                              mat, 
                                              torch.zeros(4, right, dtype = torch.double)), dim = 1)), dim = 0)
            left += self.n_state
            right -= self.n_state
 

        ctrl_mat = torch.zeros(self._Ns*self._Nt*self.n_state, self._Ns*self.n_ctrl)
        for i in range(self._Ns):
            ctrl_mat[(self.n_state*self._Nt*i):(self.n_state*self._Nt*i + self.n_state), (self.n_ctrl*i):(self.n_ctrl*i+2)] = -self._exp_At_B
            #ctrl_mat[(self.n_state*self._Nt*i):(self.n_state*self._Nt*i + self.n_state), (self.n_ctrl*i):(self.n_ctrl*i+2)] = -self._B

        self._eq_cont_mat = torch.cat((state_mat, ctrl_mat), dim = 1)
        #np.savetxt('eq_mat.txt', self._eq_cont_mat.numpy(), fmt="%.3f")

    def equality_contraint_vec(self, x_0, Tr): #already batched
        self._eq_cont_vec = torch.zeros(self._batch, 4*self._Ns*self._Nt, dtype = torch.double)
        exp_ATr = torch.linalg.matrix_exp(self._A.unsqueeze(0).repeat(self._batch, 1, 1)*(Tr).unsqueeze(1).unsqueeze(1))
        self._eq_cont_vec[:, 0:4] = torch.matmul(exp_ATr, x_0.unsqueeze(2)).squeeze()
        #np.savetxt('eq_vec.txt', self._eq_cont_vec.numpy(), fmt="%.3f")

    def cost_mat(self): #not batched
        Qrunning = 2*torch.tensor([[self._px, 0, 0, 0],
                                   [ 0, self._py, 0, 0],
                                   [ 0, 0, self._pLx, 0],
                                   [ 0, 0, 0, self._pLy]], dtype = torch.double)
        Qterminal = 100*Qrunning

        self._cost_mat = torch.diag(self.eps*torch.ones(self._Ns*self._Nt*self.n_state + self._Ns*self.n_ctrl, dtype = torch.double))
        for i in range(self._Ns - 1):
            self._cost_mat[((i+1)*self._Nt - 1)*self.n_state:(i+1)*self._Nt*self.n_state, 
                           ((i+1)*self._Nt - 1)*self.n_state:(i+1)*self._Nt*self.n_state] = Qrunning

        self._cost_mat[(self._Ns*self._Nt - 1)*self.n_state:self._Ns*self._Nt*self.n_state, 
                       (self._Ns*self._Nt - 1)*self.n_state:self._Ns*self._Nt*self.n_state] = Qterminal
        #np.savetxt('cost_mat.txt', self._cost_mat.numpy(), fmt="%.3f")
   
    def cost_vec(self, idx_plus, idx_minus): #batched
        # vector of size (Ns*Nt)*4 + Ns*2
        self.getCostvec(idx_plus, idx_minus)

        self._cost_vec = torch.zeros(self._batch, self._Ns*self._Nt*self.n_state + self._Ns*self.n_ctrl, dtype = torch.double)
        for i in range(self._Ns):
            self._cost_vec[:, ((i+1)*self._Nt - 1)*self.n_state:(i+1)*self._Nt*self.n_state,] = self.q[i, :, 0:4]

        #np.savetxt('cost_vec.txt', self._cost_vec.numpy(), fmt="%.3f")

    def ineq_mat(self): #not batched
        #neq = ---
        mat1 = torch.zeros(self._Ns*self._Nt*2 + self._Ns*self.n_ctrl,
                           self._Ns*self._Nt*self.n_state + self._Ns*self.n_ctrl, dtype = torch.double)
        for i in range(self._Ns*self._Nt):
            mat1[2*i    , i*self.n_state] = 1.
            mat1[2*i + 1, i*self.n_state + 1] = 1.
        for i in range(self._Ns):
            mat1[2*self._Ns*self._Nt + 2*i, self.n_state*self._Ns*self._Nt+ 2*i] = 1.
            mat1[2*self._Ns*self._Nt + 2*i+1, self.n_state*self._Ns*self._Nt+2*i+1] = 1.

        self._ineq_cont_mat = torch.cat((mat1, -mat1), dim = 0)
        #np.savetxt('ineq_mat.txt', self._ineq_cont_mat.numpy(), fmt="%.3f")

    
    def ineq_vec(self, idx_plus, idx_minus):
        self.get_u_bounds(idx_plus, idx_minus)
        self._ineq_cont_vec = torch.zeros(self._batch, 2*(self._Ns*self._Nt*2 + self._Ns*self.n_ctrl), dtype = torch.double)

        if (self._x_mech_limit > self._xc_slip_limit):
            self._x_bound = self._xc_slip_limit
        else:
            self._x_bound = self._x_mech_limit
        
        #U BOUNDS
        for i in range(self._Ns):
            self._ineq_cont_vec[:, self._Ns*self._Nt*2+2*i:
                                   self._Ns*self._Nt*2+2*(i+1)] = self.u_upper[i, :]
            self._ineq_cont_vec[:, 2*self._Ns*self._Nt*2 + self._Ns*self.n_ctrl+2*i:
                                   2*self._Ns*self._Nt*2 + self._Ns*self.n_ctrl+2*(i+1)] = -self.u_lower[i,:]

        for i in range(self._Ns*self._Nt):
            self._ineq_cont_vec[:, 2*i] = self._x_bound
            self._ineq_cont_vec[:, 2*i + 1] = self._yc_slip_limit
            self._ineq_cont_vec[:, self._Ns*self._Nt*2 + self._Ns*self.n_ctrl + 2*i] = self._x_bound
            self._ineq_cont_vec[:, self._Ns*self._Nt*2 + self._Ns*self.n_ctrl + 2*i + 1] = self._yc_slip_limit

        tensor_ = torch.arange(0,self._Ns*self._Nt*self.n_state + self._Ns*self.n_ctrl).to(dtype = torch.double)
        a = torch.matmul(self._ineq_cont_mat, tensor_)
        #np.savetxt("a.txt", a, fmt = "%.1f")
        #np.savetxt("ineq_vec.txt", self._ineq_cont_vec.t(), fmt = "%.3f")

        
        






    

    

    






