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
        self._Ts        = AlipParams.TS        * torch.ones(self._n_batch, dtype = torch.double)
        self._Lx_offset = AlipParams.LX_OFFSET * torch.ones(self._n_batch, dtype = torch.double)
        self._Ly_des    = AlipParams.LY_DES    * torch.ones(self._n_batch, dtype = torch.double)
        self._rf_z_MAX  = AlipParams.RF_Z_MAX  * torch.ones(self._n_batch, dtype = torch.double)
        self._rf_z_max  = 1e-4                 * torch.ones(self._n_batch, dtype = torch.double)
        self._des_com_yaw = AlipParams.COM_YAW * torch.ones(self._n_batch, dtype = torch.double)

        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()

    def first_visit(self):
        self._state_machine_start_time = self._sp.curr_time * torch.ones(self._n_batch, dtype = torch.double)
        self._trajectory_manager.initializeOri()



    def new_step(self, ids):
        print(ids)
        self._trajectory_manager.stance_leg(self._stance_leg[ids], ids)
        self._state_machine_start_time[ids] = self._sp.curr_time * torch.ones(len(ids), dtype = torch.double)
        self._state_machine_time = self._sp.curr_time - self._state_machine_start_time

        self._Tr = self._Ts - self._state_machine_time
        print("setter")
        self._trajectory_manager.des_com_yaw(self._des_com_yaw[ids], ids)

        self._trajectory_manager.setNewOri(ids) #TODO: TRAJECTORY FOR ORI
        torso_ori = self._trajectory_manager.des_torso_rot[ids]

        com_pos = self._robot.get_com_pos()[ids]
        com_vel = self._robot.get_com_lin_vel()[ids]
        rfoot_pos = self._robot.get_link_iso("r_foot_contact")[ids, 0:3, 3]
        lfoot_pos = self._robot.get_link_iso("l_foot_contact")[ids, 0:3, 3]
        self._swfoot_end = self._alip_mpc.solve_inertia_coor(self._stance_leg[ids], self._Lx_offset[ids], self._Ly_des[ids], self._Tr[ids], torso_ori,
                                                             com_pos, com_vel, lfoot_pos, rfoot_pos)

        self._trajectory_manager.generateSwingFtraj(self._state_machine_time[ids], self._Tr[ids], self._swfoot_end[ids], ids)

        #change contact and reaction forces
        new_rf_z_max_rfoot = torch.zeros(self._n_batch, dtype = torch.double)
        new_rf_z_max_lfoot = torch.zeros(self._n_batch, dtype = torch.double)
        b_lf_contact_h = self._sp.b_lf_contact
        b_rf_contact_h = self._sp.b_rf_contact

        rst_id = torch.nonzero(self._stance_leg[ids] == 1).squeeze().tolist()
        new_rf_z_max_rfoot[rst_id] = self._rf_z_MAX[rst_id]
        new_rf_z_max_lfoot[rst_id] = self._rf_z_max[rst_id]
        for i in rst_id:
            b_lf_contact_h[i] = False  
            b_rf_contact_h[i] = True

        lst_id = torch.nonzero(self._stance_leg[ids] ==-1).squeeze().tolist()
        new_rf_z_max_lfoot[lst_id] = self._rf_z_MAX[lst_id]
        new_rf_z_max_rfoot[lst_id] = self._rf_z_max[lst_id]
        for i in lst_id:
            b_lf_contact_h[i] = True
            b_rf_contact_h[i] = False
                
        self._sp.b_lf_contact = b_lf_contact_h
        self._sp.b_rf_contact = b_rf_contact_h
        #0 will be for rfoot_contact
        #1 will be for lfoot_contact
        self._tci_container.contact_list[0].rf_z_max = new_rf_z_max_rfoot 
        self._tci_container.contact_list[1].rf_z_max = new_rf_z_max_lfoot


    def one_step(self, ids): #in the controller
        self._state_machine_time = self._sp.curr_time - self._state_machine_start_time
        t = self._state_machine_time + self._Tr - self._Ts
        self._trajectory_manager.updateDesired(t, ids)

    def switchLeg(self, new_step_list):
        print("Switch", self._sp.curr_time, self._sp.curr_time - self._state_machine_start_time)
        indices = torch.nonzero(self._sp.curr_time - self._state_machine_start_time > 0.5*self._Ts)
        rfoot_z = self._robot.get_link_iso("r_foot_contact")[:, 2, 3]
        lfoot_z = self._robot.get_link_iso("l_foot_contact")[:, 2, 3]
        rfoot_rf_max = self._tci_container.contact_list[0].rf_z_max
        lfoot_rf_max = self._tci_container.contact_list[1].rf_z_max

        #res = torch.zeros(self._n_batch)
        for i in indices:
            if self._stance_leg[i] == 1 and lfoot_z[i] < 0.0005:
                self._stance_leg[i] = -1
                lfoot_rf_max[i] = AlipParams.RF_Z_MAX
                self._state_machine_start_time = self._sp.curr_time
                print("\n ---------------- \n", "Switch_leg ", i, "\n -------------------- \n")
                new_step_list[i] = 3
                if self._b_data_save:
                    self._data_saver.add('leg_switch_time', self._sp.curr_time)

            elif self._stance_leg[i] == -1 and rfoot_z[i] < 0.0005:
                self._stance_leg[i] = 1
                rfoot_rf_max[i] = AlipParams.RF_Z_MAX
                self._state_machine_start_time = self._sp.curr_time
                print("\n ---------------- \n", "Switch_leg ", i, "\n -------------------- \n")
                new_step_list[i] == 3
                if self._b_data_save:
                    self._data_saver.add('leg_switch_time', self._sp.curr_time)
        print(self._stance_leg)
        self._tci_container.contact_list[0].rf_z_max = rfoot_rf_max
        self._tci_container.contact_list[1].rf_z_max = lfoot_rf_max
        return new_step_list


    def end_of_state(self):
        pass
    def get_next_state(self):
        pass
    def last_visit(self):
        pass
