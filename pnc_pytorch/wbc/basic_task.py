from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
import numpy as np


from util import util
from util import orbit_util
from pnc_pytorch.wbc.task import Task
from pnc_pytorch.data_saver import DataSaver

def printvar(a, b):
    print(a, "\n", b, " shape" , b.shape, " | type", b.dtype, "\n")


class BasicTask(Task):
    def __init__(self, robot, task_type, dim, pos_dim, n_batch, target_id=None, data_save=False):
        super(BasicTask, self).__init__(robot, dim, pos_dim, n_batch)

        self._target_id = target_id
        self._task_type = task_type
        self._b_data_save = data_save

        if self._b_data_save:
            self._data_saver = DataSaver()

    @property
    def target_id(self):
        return self._target_id

    def update_cmd(self):#TODO: check batched (b)
                        #joint_positions, joint_velocities, pos_des
                        #get_link_iso
                        #get_link_vel
                        #get_com_pos
        if self._task_type == "JOINT":
            #TODO: not sure how the batched will work
            pos = self._robot.joint_positions  

            self._pos_err = self._pos_des - pos 

            vel_act = self._robot.joint_velocities 

            if self._b_data_save:
                self._data_saver.add('joint_pos_des', self._pos_des.clone().detach())
                self._data_saver.add('joint_vel_des', self._vel_des.clone().detach())
                self._data_saver.add('joint_pos', pos.clone().detach())
                self._data_saver.add('joint_vel', vel_act.clone().detach())
                self._data_saver.add('w_joint', self._w_hierarchy.clone().detach())
        elif self._task_type == "SELECTED_JOINT":
            pos = self._robot.joint_positions[:, self._robot.get_joint_idx(
                self._target_id)]

            self._pos_err = self._pos_des - pos

            vel_act = self._robot.joint_velocities[:, self._robot.get_joint_idx(
                self._target_id)]

            if self._b_data_save:
                self._data_saver.add('selected_joint_pos_des',
                                     self._pos_des.clone().detach())
                self._data_saver.add('selected_joint_vel_des',
                                     self._vel_des.clone().detach())
                self._data_saver.add('selected_joint_pos', pos.clone().detach())
                self._data_saver.add('selected_joint_vel', vel_act.clone().detach())
                self._data_saver.add('w_selected_joint', self._w_hierarchy.clone().detach())
        elif self._task_type == "LINK_XYZ":
            pos = self._robot.get_link_iso(self._target_id)[:, 0:3, 3]

            self._pos_err = self._pos_des - pos
        
            vel_act = self._robot.get_link_vel(self._target_id)[:, 3:6]

            if self._b_data_save:
                self._data_saver.add(self._target_id + '_pos_des',
                                     self._pos_des.clone())
                self._data_saver.add(self._target_id + '_vel_des',
                                     self._vel_des.clone())
                self._data_saver.add(self._target_id + '_pos', pos.clone())
                self._data_saver.add(self._target_id + '_vel', vel_act.clone())
                self._data_saver.add('w_' + self._target_id, self._w_hierarchy)
        elif self._task_type == "LINK_ORI":
            quat_act = orbit_util.convert_quat(orbit_util.quat_from_matrix(self._robot.get_link_iso(self._target_id)[:, 0:3, 0:3]))
            quat_act_temp_h = util.prevent_quat_jump_pytorch(self._pos_des,
                                                             quat_act)
                                                             
            quat_err = util.quat_mul_xyzw(self._pos_des, util.quat_inv_xyzw(quat_act_temp_h))

            self._pos_err = util.quat_to_exp_pytorch(quat_err)
            vel_act = self._robot.get_link_vel(self._target_id)[:, 0:3]

            if self._b_data_save:
                self._data_saver.add(self._target_id + '_quat_des',
                                     self._pos_des)
                self._data_saver.add(self._target_id + '_ang_vel_des',
                                     self._vel_des.clone())
                self._data_saver.add(self._target_id + '_quat',
                                     quat_act)
                self._data_saver.add(self._target_id + '_ang_vel',
                                     vel_act.clone())
                self._data_saver.add('w_' + self._target_id + "_ori",
                                     self._w_hierarchy)
                self._data_saver.add(self._target_id + "_quat_err",
                                     self._pos_err)
        elif self._task_type == "COM":
            pos = self._robot.get_com_pos()  

            self._pos_err = self._pos_des - pos

            vel_act = self._robot.get_com_lin_vel()

            if self._b_data_save:
                self._data_saver.add(self._target_id + '_pos_des',
                                     self._pos_des.clone())
                self._data_saver.add(self._target_id + '_vel_des',
                                     self._vel_des.clone())
                self._data_saver.add(self._target_id + '_pos', pos.clone())
                self._data_saver.add(self._target_id + '_vel', vel_act.clone())
                self._data_saver.add('w_' + self._target_id, self._w_hierarchy)
        else:
            raise ValueError

        """
        print("BASIC Task")
        printvar("acc", self._acc_des)
        printvar("kp", self._kp)
        printvar("kd", self._kd)
        printvar("pos err", self._pos_err)
        printvar("vel des", self._vel_des)
        printvar("vel_act", vel_act)
        """

        self._op_cmd = self._acc_des + self._kp.unsqueeze(0) * self._pos_err + self._kd.unsqueeze(0) * (self._vel_des - vel_act)



    def update_jacobian(self):
        if self._task_type == "JOINT":
            self._jacobian[:, :, self._robot.n_floating:self._robot.n_floating +
                           self._robot.n_a] = torch.eye(self._dim).unsqueeze(0).repeat(self.n_batch, 1, 1)
            self._jacobian_dot_q_dot = torch.zeros(self.n_batch, self._dim)
        elif self._task_type == "SELECTED_JOINT":
            for i, jid in enumerate(self._robot.get_q_dot_idx(
                    self._target_id)):
                self._jacobian[:, i, jid] = 1
            self._jacobian_dot_q_dot = torch.zeros(self.n_batch, self._dim)
        elif self._task_type == "LINK_XYZ":
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[:, 3:6, :]
            self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
                self._target_id)[:, 3:6]
        elif self._task_type == "LINK_ORI":
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[:, 0:3, :]
            self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
                self._target_id)[:, 0:3]
        elif self._task_type == "COM":
            self._jacobian = self._robot.get_com_lin_jacobian()
            self._jacobian_dot_q_dot = torch.matmul(
                self._robot.get_com_lin_jacobian_dot(),
                self._robot.get_q_dot().unsqueeze(2)).squeeze()
        else:
            raise ValueError
