import torch

from pnc_pytorch.state_machine import StateMachine
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider
from pnc_pytorch.data_saver import DataSaver
from config.draco3_alip_config import AlipParams
"""
Must read the wbc state

1 Per step
    Computes ALIP MPC solutions
    Will have the RL policy in the future
    Sends final step fos to tm 

Control fREQ
    updates
"""


class AlipLocomotion(StateMachine):
    def __init__(self, batch, id, tm, alip_mpc, tci_container, robot, data_save = False):
        self._n_batch = batch
        self._robot = robot
        self._trajectory_manager = tm
        self._alip_mpc = alip_mpc
        self._tci_container = tci_container
        self._sp = Draco3StateProvider()

        #params after will implement in set params
        self._stance_leg = AlipParams.INITIAL_STANCE_LEG * torch.ones(self._n_batch)
        self._Ts = AlipParams.TS * torch.ones(self._n_batch)
        self._Lx_offset = AlipParams.LX_OFFSET * torch.ones(self._n_batch)
        self._Ly_des = AlipParams.LY_DES * torch.ones(self._n_batch)


        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()

    def first_visit(self):
        self._state_machine_start_time = self._sp.curr_time
        self._trajectory_manager.initializeOri()



    def new_step(self):
        self._trajectory_manager.stance_leg = self._stance_leg
        self._state_machine_start_time = self._sp.curr_time
        self._state_machine_time = self._sp.curr_time - self._state_machine_start_time

        self._Tr = self._Ts - self._state_machine_time
        #self._trajectory_manager.des_com_yaw = AlipParams.COM_YAW * torch.ones(self._n_batch)

        self._trajectory_manager.setNewOri() #TODO: TRAJECTORY FOR ORI
        torso_ori = self._trajectory_manager.des_torso_rot

        com_pos = self._robot.get_com_pos()
        com_vel = self._robot.get_com_lin_vel()
        rfoot_pos = self._robot.get_link_iso("r_foot_contact")[:, 0:3, 3]
        lfoot_pos = self._robot.get_link_iso("l_foot_contact")[:, 0:3, 3]
        self._swfoot_end = self._alip_mpc.solve_inertia_coor(self._stance_leg, self._Lx_offset, self._Ly_des, self._Tr, torso_ori,
                                                             com_pos, com_vel, lfoot_pos, rfoot_pos)

        self._trajectory_manager.generateSwingFtraj(self._state_machine_time, self._Tr, self._swfoot_end)

        new_rf_z_max_rfoot = torch.zeros(self._n_batch)
        new_rf_z_max_lfoot = torch.zeros(self._n_batch)
        b_lf_contact_h = self._sp.b_lf_contact
        b_rf_contact_h = self._sp.b_rf_contact
        for i in range(self._n_batch):
            if (self._stance_leg[i] == 1):
                """
                self._sp._b_lf_contact[i] = False
                self._sp._b_rf_contact[i] = True
                """
                b_lf_contact_h[i] = False
                b_rf_contact_h[i] = True
                new_rf_z_max_rfoot[i] = AlipParams.RF_Z_MAX
                new_rf_z_max_lfoot[i] = 1e-4
            else:
                """
                self._sp._b_rf_contact[i] = False
                self._sp._b_lf_contact[i] = True
                """
                b_rf_contact_h[i] = False
                b_lf_contact_h[i] = True
                new_rf_z_max_lfoot[i] = AlipParams.RF_Z_MAX
                new_rf_z_max_rfoot[i] = 1e-4
        self._sp.b_lf_contact = b_lf_contact_h
        self._sp.b_rf_contact = b_rf_contact_h
        #0 will be for rfoot_contact
        #1 will be for lfoot_contact
        self._tci_container.contact_list[0].rf_z_max = new_rf_z_max_rfoot 
        self._tci_container.contact_list[1].rf_z_max = new_rf_z_max_lfoot


    def one_step(self): #in the controller
        self._state_machine_time = self._sp.curr_time - self._state_machine_start_time
        t = self._state_machine_time + self._Tr - self._Ts
        self._trajectory_manager.updateDesired(t)

    def switchLeg(self):
        #print("Switch", self._sp.curr_time, self._sp.curr_time - self._state_machine_start_time)
        indices = torch.nonzero(self._sp.curr_time - self._state_machine_start_time > 0.5*self._Ts)
        rfoot_z = self._robot.get_link_iso("r_foot_contact")[:, 2, 3]
        lfoot_z = self._robot.get_link_iso("l_foot_contact")[:, 2, 3]
        rfoot_rf_max = self._tci_container.contact_list[0].rf_z_max
        lfoot_rf_max = self._tci_container.contact_list[1].rf_z_max
        res = False
        for i in indices:
            if self._stance_leg[i] == 1 and lfoot_z[i] < 0.0005:
                self._stance_leg[i] = -1
                lfoot_rf_max[i] = AlipParams.RF_Z_MAX
                self._state_machine_start_time = self._sp.curr_time
                print("\n ---------------- \n", "Switch_leg ", i, "\n -------------------- \n")
                res = True
                if self._b_data_save:
                    self._data_saver.add('leg_switch_time', self._sp.curr_time)


            elif self._stance_leg[i] == -1 and rfoot_z[i] < 0.0005:
                self._stance_leg[i] = 1
                rfoot_rf_max[i] = AlipParams.RF_Z_MAX
                self._state_machine_start_time = self._sp.curr_time
                print("\n ---------------- \n", "Switch_leg ", i, "\n -------------------- \n")
                res = True
                if self._b_data_save:
                    self._data_saver.add('leg_switch_time', self._sp.curr_time)

        self._tci_container.contact_list[0].rf_z_max = rfoot_rf_max
        self._tci_container.contact_list[1].rf_z_max = lfoot_rf_max
        return res


    def end_of_state(self):
        pass
    def get_next_state(self):
        pass
    def last_visit(self):
        pass
