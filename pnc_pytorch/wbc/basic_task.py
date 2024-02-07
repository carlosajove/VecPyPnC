from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
import numpy as np

from util import util
from pnc_pytorch.wbc.task import Task
from pnc_pytorch.data_saver import DataSaver


class BasicTask(Task):
    def __init__(self, robot, task_type, dim, n_batch, target_id=None, data_save=False):
        super(BasicTask, self).__init__(robot, dim, n_batch)

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
            #TODO: robot
            pos = self._robot.joint_positions  
            pos = torch.from_numpy(pos).expand(self.n_batch, -1)

            self._pos_err = self._pos_des - pos 

            vel_act = self._robot.joint_velocities 
            vel_act = torch.from_numpy(vel_act).expand(self.n_batch, -1)

            if self._b_data_save:
                self._data_saver.add('joint_pos_des', self._pos_des.clone().detach())
                self._data_saver.add('joint_vel_des', self._vel_des.clone().detach())
                self._data_saver.add('joint_pos', pos.clone().detach())
                self._data_saver.add('joint_vel', vel_act.clone().detach())
                self._data_saver.add('w_joint', self._w_hierarchy.clone().detach())
        elif self._task_type == "SELECTED_JOINT":
            pos = self._robot.joint_positions[self._robot.get_joint_idx(
                self._target_id)]
            pos = torch.from_numpy(pos).expand(self.n_batch, -1)

            self._pos_err = self._pos_des - pos

            vel_act = self._robot.joint_velocities[self._robot.get_joint_idx(
                self._target_id)]
            vel_act = torch.from_numpy(vel_act).expand(self.n_batch, -1)

            if self._b_data_save:
                self._data_saver.add('selected_joint_pos_des',
                                     self._pos_des.clone().detach())
                self._data_saver.add('selected_joint_vel_des',
                                     self._vel_des.clone().detach())
                self._data_saver.add('selected_joint_pos', pos.clone().detach())
                self._data_saver.add('selected_joint_vel', vel_act.clone().detach())
                self._data_saver.add('w_selected_joint', self._w_hierarchy.clone().detach())
        elif self._task_type == "LINK_XYZ":
            """
            pos = self._robot.get_link_iso(self._target_id)[:, 0:3, 3]
            """
            pos = self._robot.get_link_iso(self._target_id)[0:3, 3]
            pos = torch.from_numpy(pos).expand(self.n_batch, -1)

            self._pos_err = self._pos_des - pos
            """
            vel_act = self._robot.get_link_vel(self._target_id)[:, 3:6]
            """
            vel_act = self._robot.get_link_vel(self._target_id)[3:6]

            vel_act = torch.from_numpy(vel_act).expand(self.n_batch, -1)

            if self._b_data_save:
                self._data_saver.add(self._target_id + '_pos_des',
                                     self._pos_des.clone())
                self._data_saver.add(self._target_id + '_vel_des',
                                     self._vel_des.clone())
                self._data_saver.add(self._target_id + '_pos', pos.clone())
                self._data_saver.add(self._target_id + '_vel', vel_act.clone())
                self._data_saver.add('w_' + self._target_id, self._w_hierarchy)
        elif self._task_type == "LINK_ORI":
            """--------------------------
            TODO: use orbit functions instead of scipy rotations
            -----------------------------"""
            pos_des_np = self._pos_des[0, :].numpy()
            quat_des = R.from_quat(pos_des_np)
            """
            quat_act = R.from_matrix(
                self._robot.get_link_iso(self._target_id)[:, 0:3, 0:3])
            """
            quat_act = R.from_matrix(self._robot.get_link_iso(self._target_id)[0:3, 0:3])
            quat_des_temp = quat_des.as_quat()
            quat_act_temp = quat_act.as_quat()
            quat_act_temp = util.prevent_quat_jump(quat_des_temp,
                                                   quat_act_temp)
            quat_act = R.from_quat(quat_act_temp)
            quat_err = R.from_matrix(
                np.dot(quat_des.as_matrix(),
                       quat_act.as_matrix().transpose())).as_quat()
            self._pos_err = util.quat_to_exp(quat_err)

            self._pos_err = torch.from_numpy(self._pos_err).expand(self.n_batch, -1)

            vel_act = self._robot.get_link_vel(self._target_id)[0:3]
            vel_act = torch.from_numpy(vel_act).expand(self.n_batch, -1)

            if self._b_data_save:
                self._data_saver.add(self._target_id + '_quat_des',
                                     quat_des.as_quat())
                self._data_saver.add(self._target_id + '_ang_vel_des',
                                     self._vel_des.clone())
                self._data_saver.add(self._target_id + '_quat',
                                     quat_act.as_quat())
                self._data_saver.add(self._target_id + '_ang_vel',
                                     vel_act.clone())
                self._data_saver.add('w_' + self._target_id + "_ori",
                                     self._w_hierarchy)
                self._data_saver.add(self._target_id + "_quat_err",
                                     self._pos_err)
        elif self._task_type == "COM":
            pos = self._robot.get_com_pos()  
            pos = torch.from_numpy(pos).expand(self.n_batch, -1)

            self._pos_err = self._pos_des - pos

            vel_act = self._robot.get_com_lin_vel()
            vel_act = torch.from_numpy(vel_act).expand(self.n_batch, -1)

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
            """
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[:, 3:6, :]
            self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
                self._target_id)[:, 3:6]
            """
            j = self._robot.get_link_jacobian(self._target_id)[3:6, :]
            jqdq = self._robot.get_link_jacobian_dot_times_qdot(self._target_id)[3:6]
            self._jacobian = torch.from_numpy(j).expand(self.n_batch, -1, -1)
            self._jacobian_dot_q_dot = torch.from_numpy(jqdq).expand(self.n_batch, -1)
        elif self._task_type == "LINK_ORI":
            """
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[:, 0:3, :]
            self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
                self._target_id)[:, 0:3]
            """
            j = self._robot.get_link_jacobian(self._target_id)[0:3, :]
            jqdq = self._robot.get_link_jacobian_dot_times_qdot(self._target_id)[0:3]
            self._jacobian = torch.from_numpy(j).expand(self.n_batch, -1, -1)
            self._jacobian_dot_q_dot = torch.from_numpy(jqdq).expand(self.n_batch, -1)
        elif self._task_type == "COM":
            """
            self._jacobian = self._robot.get_com_lin_jacobian()
            self._jacobian_dot_q_dot = torch.bmm(
                self._robot.get_com_lin_jacobian_dot(),
                self._robot.get_q_dot())
            """
            self._jacobian = torch.from_numpy(self._robot.get_com_lin_jacobian()).expand(self.n_batch, -1, -1)
            self._jacobian_dot_q_dot = torch.matmul(
                torch.from_numpy(self._robot.get_com_lin_jacobian_dot()).expand(self.n_batch, -1, -1),
                torch.from_numpy(self._robot.get_q_dot()).expand(self.n_batch, -1).unsqueeze(2)).squeeze()
        else:
            raise ValueError
