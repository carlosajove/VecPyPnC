import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from config.draco3_alip_config import WalkingState
from pnc_pytorch.state_machine import StateMachine
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider


class DoubleSupportStand(StateMachine):
    def __init__(self, batch, id, tm, fm, robot, verbose = False):
        super(DoubleSupportStand, self).__init__(id, robot, verbose)
        self._n_batch = batch
        self._trajectory_managers = tm
        self._force_managers = fm   
        self._end_time = 0.
        self._rf_z_max_time = torch.zeros(self._n_batch)
        self._com_height_des = torch.zeros(self._n_batch)
        self._start_time = 0.
        self._sp = Draco3StateProvider()

    @property
    def end_time(self):
        return self._end_time

    @property
    def rf_z_max_time(self):
        return self.rf_z_max_time

    @property
    def com_height_des(self):
        return self.com_height_des

    @end_time.setter
    def end_time(self, val):
        self._end_time = val

    @rf_z_max_time.setter
    def rf_z_max_time(self, val):
        self._rf_z_max_time = val

    @com_height_des.setter
    def com_height_des(self, val):
        self._com_height_des = val

    def first_visit(self):
        if self._verbose:
            print("[WalkingState] STAND")
        self._start_time = self._sp.curr_time 

        # Initialize CoM Trajectory
        lfoot_iso = self._robot.get_link_iso("l_foot_contact")
        rfoot_iso = self._robot.get_link_iso("r_foot_contact")



        com_pos_des = (lfoot_iso[:, 0:3, 3] + rfoot_iso[:, 0:3, 3]) / 2.0
        com_pos_des[:, 2] = self._com_height_des

        #TODO: change when rots
        np_lfoot_iso = lfoot_iso[0].numpy()
        np_rfoot_iso = rfoot_iso[0].numpy()
        base_quat_slerp = Slerp(
            [0, 1], R.from_matrix([np_lfoot_iso[0:3, 0:3], np_rfoot_iso[0:3, 0:3]]))
        base_quat_des = base_quat_slerp(0.5).as_quat()

        base_quat_des = torch.from_numpy(base_quat_des).expand(self._n_batch, -1)

        self._trajectory_managers[
            "floating_base"].initialize_floating_base_interpolation_trajectory(
                self._sp.curr_time * torch.ones(self._n_batch), self._end_time, com_pos_des, base_quat_des)

        # Initialize Reaction Force Ramp to Max
        for fm in self._force_managers.values():
            fm.initialize_ramp_to_max(self._sp.curr_time, self._rf_z_max_time)

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Floating Base Task
        self._trajectory_managers[
            "floating_base"].update_floating_base_desired(self._sp.curr_time)
        # Update Foot Task
        self._trajectory_managers["alip_tm"].use_both_current()

        # Update Max Normal Reaction Force
        for fm in self._force_managers.values():
            fm.update_ramp_to_max(self._sp.curr_time)

    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time > self._end_time:
            return True
        else:
            return False

    def get_next_state(self):
        return WalkingState.BALANCE
