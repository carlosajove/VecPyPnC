import numpy as np
import torch 
from config.draco3_alip_config import PnCConfig
from util import util
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider
from util import orbit_util

class Draco3StateEstimator(object):
    def __init__(self, robot, n_batch):
        super(Draco3StateEstimator, self).__init__()
        self._robot = robot
        self._n_batch = n_batch
        self._sp = Draco3StateProvider(self._robot, n_batch)

    def initialize(self, sensor_data):
        self._sp.nominal_joint_pos = sensor_data["joint_pos"]

    def update(self, sensor_data):

        # Update Encoders
        self._robot.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"])

        # Update Contact Info
        # TODO: change when new interface
        self._sp.b_rf_contact = [sensor_data["b_rf_contact"]] * self._n_batch
        self._sp.b_lf_contact = [sensor_data["b_lf_contact"]] * self._n_batch


    def inertia_to_com_torso_coor(self):
        com_pos = self._robot.get_com_pos()
        com_vel = self._robot.get_com_lin_vel()
        torso_matrix = self._robot.get_link_iso("torso_com_link")[:, 0:3, 0:3]
        torso_quat = orbit_util.quat_from_matrix(torso_matrix)

        rfoot_pos = self._robot.get_link_iso("r_foot_contact")[:, 0:3, 3]
        lfoot_pos = self._robot.get_link_iso("l_foot_contact")[:, 0:3, 3]

        stleg_pos = torch.where(self._sp.stance_leg.unsqueeze(1) == 1, rfoot_pos, lfoot_pos)

        com_pos_stleg = com_pos - stleg_pos

        #stleg_pos = stleg_pos.to(torso_ori.dtype)
        com_pos_stleg_torso_ori = orbit_util.quat_rotate_inverse(torso_quat, com_pos_stleg)
        com_vel_torso_ori = orbit_util.quat_rotate_inverse(torso_quat, com_vel)
        

        L = self._sp.mass*torch.linalg.cross(com_pos_stleg_torso_ori, com_vel_torso_ori)

        #could also add com_vel
        self._sp.com_pos_stance_frame = com_pos_stleg_torso_ori
        self._sp.L_stance_frame = L
        self._sp.stleg_pos = stleg_pos
        #TODO remove squeeze
        self._sp.torso_roll_pitch_yaw = torch.cat(orbit_util.euler_xyz_from_quat(torso_quat)).unsqueeze(0)
