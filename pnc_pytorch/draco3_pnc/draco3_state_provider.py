from collections import OrderedDict

import numpy as np
import torch

from util import orbit_util
from util import util

from config.draco3_alip_config import AlipParams

class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Draco3StateProvider(metaclass=MetaSingleton):
    def __init__(self, robot, batch):
        self._robot = robot
        self._batch = batch
        #TODO: check nominal joint pos ordered Dict
        self._nominal_joint_pos = OrderedDict() 
        self._state = 0
        self._prev_state = 0
        self._curr_time = 0
        """
        self._b_rf_contact = True
        self._b_lf_contact = True
        """
        self._b_rf_contact = [True] * self._batch
        self._b_lf_contact = [True] * self._batch  

        self._Ts = AlipParams.TS * torch.ones(self._batch, dtype = torch.double)
        self._mass = AlipParams.MASS
        self._stance_leg = AlipParams.INITIAL_STANCE_LEG * torch.ones(self._batch, dtype = torch.double)
        self._Lx_offset = AlipParams.LX_OFFSET * torch.ones(self._batch, dtype = torch.double)
        self._Ly_des = AlipParams.LY_DES * torch.ones(self._batch, dtype = torch.double)
        self._des_com_yaw = AlipParams.COM_YAW * torch.ones(self._batch, dtype = torch.double)
        self._Tr = torch.clone(self._Ts)
        

    def update_command(self):
        config = util.read_config('/home/carlos/Desktop/Austin/SeungHyeonProject/PyPnc_pytorch/config/draco3_alip_config_dyn.ini')
        PARAMS = config['Parameters']
        #self._Ts        = PARAMS.getfloat('TS')        * torch.ones(self._batch, dtype = torch.double)
        self._Lx_offset = PARAMS.getfloat('LX_OFFSET') * torch.ones(self._batch, dtype = torch.double)
        self._Ly_des    = PARAMS.getfloat('LY_DES')    * torch.ones(self._batch, dtype = torch.double)
        self._des_com_yaw = PARAMS.getfloat('COM_YAW')    * torch.ones(self._batch, dtype = torch.double) 

        print("heyy", self._des_com_yaw)       


    def get_rl_observation(self):
        #TODO: right know works for one, might need to change to update the obs when multiple robots with different tempos in sim
        #stance_leg_ids #batch x 1
        #Lx_offset_des  #batch x 1
        #Ly_des         #batch x 1
        #des_com_yaw    #batch x 1
        #Ts             #batch x 1 for now constant
        #Tr             #batch x 1 for now constant = Ts
        #com coordinates
        #com_wrt_stance #batch x 3
        #actual_com_L   #batch x 3
        #stance leg     #batch x 3    position in world frame, so we substract it to base in sensor data
        #TOTAL SIZE     #batch x 15

        #TODO: have a base and com may be redundant?


        com_pos_stance_frame, L_stance_frame, stleg_pos = self.inertia_to_com_torso_coor()

        rl_wbc_obs = torch.cat((self._stance_leg.unsqueeze(1), 
                                self._Lx_offset.unsqueeze(1), 
                                self._Ly_des.unsqueeze(1), 
                                self._des_com_yaw.unsqueeze(1), 
                                self._Ts.unsqueeze(1), 
                                self._Tr.unsqueeze(1), 
                                com_pos_stance_frame, 
                                L_stance_frame, 
                                stleg_pos), dim = 1)
        return rl_wbc_obs


    def inertia_to_com_torso_coor(self):
        com_pos = self._robot.get_com_pos()
        com_vel = self._robot.get_com_lin_vel()
        torso_matrix = self._robot.get_link_iso("torso_com_link")[:, 0:3, 0:3]
        torso_quat = orbit_util.quat_from_matrix(torso_matrix)

        rfoot_pos = self._robot.get_link_iso("r_foot_contact")[:, 0:3, 3]
        lfoot_pos = self._robot.get_link_iso("l_foot_contact")[:, 0:3, 3]

        stleg_pos = torch.where(self._stance_leg.unsqueeze(1) == 1, rfoot_pos, lfoot_pos)

        com_pos_stleg = com_pos - stleg_pos

        #stleg_pos = stleg_pos.to(torso_ori.dtype)
        com_pos_stleg_torso_ori = orbit_util.quat_rotate_inverse(torso_quat, com_pos_stleg)
        com_vel_torso_ori = orbit_util.quat_rotate_inverse(torso_quat, com_vel)
        

        L = self._mass*torch.linalg.cross(com_pos_stleg_torso_ori, com_vel_torso_ori)

        #could also add com_vel
        return com_pos_stleg_torso_ori, L, stleg_pos 


    
    @property
    def Ts(self):
        return self._Ts

    @Ts.setter
    def Ts(self, value):
        self._Ts = value
    
    @property
    def stance_leg(self):
        return self._stance_leg

    @stance_leg.setter
    def stance_leg(self, value):
        self._stance_leg = value

    @property
    def Lx_offset(self):
        return self._Lx_offset

    @Lx_offset.setter
    def Lx_offset(self, value):
        self._Lx_offset = value

    @property
    def Ly_des(self):
        return self._Ly_des

    @Ly_des.setter
    def Ly_des(self, value):
        self._Ly_des = value
    
    @property
    def des_com_yaw(self):
        return self._des_com_yaw
    
    @des_com_yaw.setter
    def des_com_yaw(self, value):
        return self._des_com_yaw

    @property
    def nominal_joint_pos(self):
        return self._nominal_joint_pos

    @property
    def state(self):
        return self._state

    @property
    def prev_state(self):
        return self._prev_state

    @property
    def dcm(self):
        return self._dcm

    @dcm.setter
    def dcm(self, value):
        self._dcm = value

    @property
    def prev_dcm(self):
        return self._prev_dcm

    @prev_dcm.setter
    def prev_dcm(self, value):
        self._prev_dcm = value

    @property
    def dcm_vel(self):
        return self._dcm_vel

    @dcm_vel.setter
    def dcm_vel(self, value):
        self._dcm_vel = value

    @prev_state.setter
    def prev_state(self, value):
        self._prev_state = value

    @property
    def curr_time(self):
        return self._curr_time

    @nominal_joint_pos.setter
    def nominal_joint_pos(self, val):
        assert self._robot.n_a == len(val.keys())
        self._nominal_joint_pos = val

    @state.setter
    def state(self, val):
        self._state = val

    @curr_time.setter
    def curr_time(self, val):
        self._curr_time = val

    @property
    def b_rf_contact(self):
        return self._b_rf_contact

    @b_rf_contact.setter
    def b_rf_contact(self, value):
        self._b_rf_contact = value

    @property
    def b_lf_contact(self):
        return self._b_lf_contact

    @b_lf_contact.setter
    def b_lf_contact(self, value):
        self._b_lf_contact = value
