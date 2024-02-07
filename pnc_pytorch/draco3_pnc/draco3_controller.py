import numpy as np
import torch

from util import util
from pnc_pytorch.data_saver import DataSaver
from config.draco3_alip_config import PnCConfig, WBCConfig
from pnc_pytorch.wbc.ihwbc.ihwbc import IHWBC


class Draco3Controller(object):
    def __init__(self, tci_container, robot, batch):
        self._tci_container = tci_container
        self._robot = robot
        self._n_batch = batch
        # Initialize WBC
        l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx = self._robot.get_q_dot_idx(
            ['l_knee_fe_jp', 'l_knee_fe_jd', 'r_knee_fe_jp', 'r_knee_fe_jd'])
        act_list = [False] * robot.n_floating + [True] * robot.n_a
        act_list[l_jd_idx] = False
        act_list[r_jd_idx] = False

        n_q_dot = len(act_list)
        n_active = torch.count_nonzero(torch.tensor(act_list))
        n_passive = n_q_dot - n_active - 6

        self._sa = torch.zeros((n_active, n_q_dot)).expand(self._n_batch, -1, -1)
        self._sv = torch.zeros((n_passive, n_q_dot)).expand(self._n_batch, -1, -1)
        j, k = 0, 0
        for i in range(n_q_dot):
            if i >= 6:
                if act_list[i]:
                    self._sa[:, j, i] = 1.
                    j += 1
                else:
                    self._sv[:, k, i] = 1.
                    k += 1
        self._sf = torch.zeros((6, n_q_dot))
        self._sf[0:6, 0:6] = torch.eye(6)
        self._sf = self._sf.expand(self._n_batch, -1, -1)

        self._ihwbc = IHWBC(self._sf, self._sa, self._sv, self._n_batch, PnCConfig.SAVE_DATA)
        
        #TODO: robot.joint_trq_limit must not be batched, might have to change is batched
        if WBCConfig.B_TRQ_LIMIT:
            self._ihwbc.trq_limit = torch.bmm(self._sa[:,:, 6:],
                                           self._robot.joint_trq_limit)
        self._ihwbc.lambda_q_ddot = WBCConfig.LAMBDA_Q_DDOT
        self._ihwbc.lambda_rf = WBCConfig.LAMBDA_RF

        self._b_first_visit = True

        if PnCConfig.SAVE_DATA:
            self._data_saver = DataSaver()

    def get_command(self):
        """
        if self._b_first_visit:
            self.first_visit()
        """

        # Dynamics properties
        mass_matrix = self._robot.get_mass_matrix()
        mass_matrix_inv = torch.linalg.inv(mass_matrix)

        coriolis = self._robot.get_coriolis()
        gravity = self._robot.get_gravity()

        """
        This will remain until robot is changed
        """
        """
        mass_matrix = torch.from_numpy(mass_matrix)
        mass_matrix_inv = torch.linalg.inv(mass_matrix)
        mass_matrix = mass_matrix.expand(self._n_batch, -1, -1)
        mass_matrix = mass_matrix_inv.expand(self._n_batch, -1, -1)
        coriolis = torch.from_numpy(coriolis).expand(self._n_batch, -1)
        gravity = torch.from_numpy(gravity).expand(self._n_batch, -1)
        """



        self._ihwbc.update_setting(mass_matrix, mass_matrix_inv, coriolis,
                                   gravity)
        # Task, Contact, and Internal Constraint Setup
        w_hierarchy_list = []
        for task in self._tci_container.task_list:
            task.update_jacobian()
            task.update_cmd()
            w_hierarchy_list.append(task.w_hierarchy)  #each task.w_hierarchy will be n_batched torch.tensor
        #self._ihwbc.w_hierarchy = torch.tensor(w_hierarchy_list)  #shape is [n_batch, #tasks]  
        self._ihwbc.w_hierarchy = torch.stack(w_hierarchy_list, dim = 1)
        assert self._ihwbc.w_hierarchy.shape[0] == self._n_batch

        for contact in self._tci_container.contact_list:
            contact.update_contact()
        for internal_constraint in self._tci_container.internal_constraint_list:
            internal_constraint.update_internal_constraint()
        # WBC commands
        joint_trq_cmd, joint_acc_cmd, rf_cmd = self._ihwbc.solve(
            self._tci_container.task_list, self._tci_container.contact_list,
            self._tci_container.internal_constraint_list)
        joint_trq_cmd = torch.bmm(self._sa[:, :, 6:].transpose(1, 2), joint_trq_cmd)
        joint_acc_cmd = torch.bmm(self._sa[:, :, 6:].transpose(1, 2), joint_acc_cmd)

        if PnCConfig.SAVE_DATA:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)


        #TODO: change when robot changed
        joint_trq_cmd = joint_trq_cmd[0].numpy()
        command = self._robot.create_cmd_ordered_dict(joint_trq_cmd)
        return command


