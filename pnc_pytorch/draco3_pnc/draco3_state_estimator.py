import numpy as np
import torch 
from config.draco3_alip_config import PnCConfig
from util import util
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider


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
