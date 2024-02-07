import numpy as np
import torch

from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class UpperBodyTrajectoryManager(object):
    def __init__(self, batch, upper_body_task, robot):
        self._n_batch = batch
        self._upper_body_task = upper_body_task
        self._robot = robot

    def use_nominal_upper_body_joint_pos(self, nominal_joint_pos):
        """
        Parameters
        ----------
        nominal_joint_pos (OrderedDict):
            Nominal joint positions
        """
        joint_pos_des = torch.tensor(
            [nominal_joint_pos[k] for k in self._upper_body_task.target_id])
        """
        Apaño change when new
        """
        joint_pos_des = joint_pos_des.expand(self._n_batch, -1)

        joint_vel_des, joint_acc_des = torch.zeros_like(
        joint_pos_des), torch.zeros_like(joint_pos_des)
        self._upper_body_task.update_desired(joint_pos_des, joint_vel_des,
                                             joint_acc_des)

    @property
    def task(self):
        return self._upper_body_task
