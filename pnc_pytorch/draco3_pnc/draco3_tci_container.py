import torch
from config.draco3_alip_config import WBCConfig, PnCConfig
from pnc_pytorch.draco3_pnc.draco3_rolling_joint_constraint import Draco3RollingJointConstraint
from pnc_pytorch.wbc.tci_container import TCIContainer
from pnc_pytorch.wbc.basic_task import BasicTask
from pnc_pytorch.wbc.basic_contact import SurfaceContact


class Draco3TCIContainer(TCIContainer):
    def __init__(self, robot, n_batch):
        super(Draco3TCIContainer, self).__init__(robot)

        self.n_batch = n_batch
        # ======================================================================
        # Initialize Task
        # ======================================================================
        # COM Task
        self._com_task = BasicTask(robot, "COM", 3, 3, self.n_batch, 'com', PnCConfig.SAVE_DATA)
        self._com_task.kp = WBCConfig.KP_COM 
        self._com_task.kd = WBCConfig.KD_COM 
        self._com_task.w_hierarchy = WBCConfig.W_COM * torch.ones(self.n_batch)

        # Torso orientation task
        self._torso_ori_task = BasicTask(robot, "LINK_ORI", 3, 4, self.n_batch,
                                         "torso_com_link", PnCConfig.SAVE_DATA)
        self._torso_ori_task.kp = WBCConfig.KP_TORSO 
        self._torso_ori_task.kd = WBCConfig.KD_TORSO 
        self._torso_ori_task.w_hierarchy = WBCConfig.W_TORSO * torch.ones(self.n_batch)

        # Upperbody joints
        upperbody_joint = [
            'neck_pitch', 'l_shoulder_fe', 'l_shoulder_aa', 'l_shoulder_ie',
            'l_elbow_fe', 'l_wrist_ps', 'l_wrist_pitch', 'r_shoulder_fe',
            'r_shoulder_aa', 'r_shoulder_ie', 'r_elbow_fe', 'r_wrist_ps',
            'r_wrist_pitch'
        ]
        self._upper_body_task = BasicTask(robot, "SELECTED_JOINT",
                                          len(upperbody_joint), len(upperbody_joint),
                                          self.n_batch, upperbody_joint, PnCConfig.SAVE_DATA)
        self._upper_body_task.kp = WBCConfig.KP_UPPER_BODY 
        self._upper_body_task.kd = WBCConfig.KD_UPPER_BODY 
        self._upper_body_task.w_hierarchy = WBCConfig.W_UPPER_BODY * torch.ones(self.n_batch)

        # Rfoot Pos Task
        self._rfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3, 3, self.n_batch,
                                         "r_foot_contact", PnCConfig.SAVE_DATA)
        self._rfoot_pos_task.kp = WBCConfig.KP_FOOT_POS
        self._rfoot_pos_task.kd = WBCConfig.KD_FOOT_POS
        self._rfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT * torch.ones(self.n_batch)

        # Lfoot Pos Task
        self._lfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3, 3, self.n_batch,
                                         "l_foot_contact", PnCConfig.SAVE_DATA)
        self._lfoot_pos_task.kp = WBCConfig.KP_FOOT_POS
        self._lfoot_pos_task.kd = WBCConfig.KD_FOOT_POS
        self._lfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT * torch.ones(self.n_batch)

        # Rfoot Ori Task
        self._rfoot_ori_task = BasicTask(robot, "LINK_ORI", 3, 4, self.n_batch,
                                         "r_foot_contact", PnCConfig.SAVE_DATA)
        self._rfoot_ori_task.kp = WBCConfig.KP_FOOT_ORI
        self._rfoot_ori_task.kd = WBCConfig.KD_FOOT_ORI
        self._rfoot_ori_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT * torch.ones(self.n_batch)

        # Lfoot Ori Task
        self._lfoot_ori_task = BasicTask(robot, "LINK_ORI", 3, 4, self.n_batch,
                                         "l_foot_contact", PnCConfig.SAVE_DATA)
        self._lfoot_ori_task.kp = WBCConfig.KP_FOOT_ORI
        self._lfoot_ori_task.kd = WBCConfig.KD_FOOT_ORI
        self._lfoot_ori_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT * torch.ones(self.n_batch)

        self._task_list = [
            self._com_task, self._torso_ori_task, self._upper_body_task,
            self._rfoot_pos_task, self._lfoot_pos_task, self._rfoot_ori_task,
            self._lfoot_ori_task
        ]

        # ======================================================================
        # Initialize Contact
        # ======================================================================
        # Rfoot Contact
        self._rfoot_contact = SurfaceContact(robot, "r_foot_contact", 0.115,
                                             0.065, 0.3, self.n_batch, True)
        self._rfoot_contact.rf_z_max = 1e-3 * torch.ones(self.n_batch) # Initial rf_z_max
        # Lfoot Contact
        self._lfoot_contact = SurfaceContact(robot, "l_foot_contact", 0.115,
                                             0.065, 0.3, self.n_batch, True)
        self._lfoot_contact.rf_z_max = 1e-3  * torch.ones(self.n_batch) # Initial rf_z_max

        #alip_locomotion requires list of size 2
        #0 for rfoot contact
        #1 for lfoot contact
        self._contact_list = [self._rfoot_contact, self._lfoot_contact]

        # ======================================================================
        # Initialize Internal Constraint
        # ======================================================================
        self._rolling_joint_constraint = Draco3RollingJointConstraint(robot, self.n_batch)
        self._internal_constraint_list = [self._rolling_joint_constraint]

    @property
    def com_task(self):
        return self._com_task

    @property
    def torso_ori_task(self):
        return self._torso_ori_task

    @property
    def upper_body_task(self):
        return self._upper_body_task

    @property
    def rfoot_pos_task(self):
        return self._rfoot_pos_task

    @property
    def lfoot_pos_task(self):
        return self._lfoot_pos_task

    @property
    def rfoot_ori_task(self):
        return self._rfoot_ori_task

    @property
    def lfoot_ori_task(self):
        return self._lfoot_ori_task

    @property
    def rfoot_contact(self):
        return self._rfoot_contact

    @property
    def lfoot_contact(self):
        return self._lfoot_contact

    @property
    def task_list(self):
        return self._task_list

    @property
    def contact_list(self):
        return self._contact_list
