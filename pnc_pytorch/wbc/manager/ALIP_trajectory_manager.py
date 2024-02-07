import torch 
from util import util
from util import interpolation

#everithing has the shape [n_batch , normal_shape]
class ALIPtrajectoryManager(object):
    def __init__(self, _n_batch, com_task, torso_ori_task, lfoot_task, lfoot_ori_task,
                rfoot_task, rfoot_ori_task, robot):
        self.n_batch = _n_batch
        #self.com_z_task = _com_z_task
        self._com_task = com_task
        self._torso_ori_task = torso_ori_task
        self._lfoot_task = lfoot_task
        self._lfoot_ori_task = lfoot_ori_task
        self._rfoot_task = rfoot_task
        self._rfoot_ori_task = rfoot_ori_task
        self._robot = robot

        assert self._rfoot_task.target_id == self._rfoot_ori_task.target_id 
        assert self._lfoot_task.target_id == self._lfoot_ori_task.target_id 
        
        self.torso_id = self._torso_ori_task.target_id
        self.lfoot_id = self._lfoot_task.target_id
        self.rfoot_id = self._rfoot_task.target_id

        self.AlipSwing2_curve = None

        #frame of reference is intertial
        self.Swingfoot_start = torch.zeros(self.n_batch, 3)
        self.Swingfoot_end = torch.zeros(self.n_batch, 3)
        self.stleg_pos = torch.zeros(self.n_batch, 3)

        self.stleg_pos_torso_ori = torch.zeros(self.n_batch, 3)  #frame of ref is torso ori 

        self.des_swfoot_pos = torch.zeros(self.n_batch, 3)
        self.des_sw_foot_vel = torch.zeros(self.n_batch, 3)
        self.des_swfoot_acc = torch.zeros(self.n_batch, 3)

        self.des_ori_lfoot = torch.tensor([0, 0, 0, 1]).unsqueeze(0).repeat(self.n_batch, 1, 1)
        self.des_ori_rfoot = torch.tensor([0, 0, 0, 1]).unsqueeze(0).repeat(self.n_batch, 1, 1)
        self.des_ori_torso = torch.tensor([0, 0, 0, 1]).unsqueeze(0).repeat(self.n_batch, 1, 1)

        self._des_torso_rot = torch.eye(3).unsqueeze(0).repeat(self.n_batch, 1, 1)
        self.des_lfoot_rot = torch.eye(3).unsqueeze(0).repeat(self.n_batch, 1, 1)
        self.des_rfoot_rot = torch.eye(3).unsqueeze(0).repeat(self.n_batch, 1, 1)
        self.des_com_yaw  = torch.zeros(self.n_batch) #rotation per step

        #indata variables mpc frame of refrence (stance foot frame of ref)
        self._stance_leg = None
        self.Ts = None

        #time vars
        self.swing_start_time = 0.

        #parameters:#set from parameters in ctrl arch
        self.swing_height = 0.05*torch.ones(self.n_batch)
        self.refzH = 0.685


        #task weights 
        """
        self._com_z_task_weight = torch.tensor([9000, 8000, 8000])
        self._com_xy_task_weight = torch.tensor([0,0])
        self._torso_ori_weight = torch.tensor([8000, 8000, 8000])
        self._swing_foot_weight = torch.tensor([8000, 8000, 8000])
        self._stance_foot_weight = torch.tensor([6000, 6000, 6000])
        self._stance_foot_ori_weight = torch.tensor([8000, 5000, 5000])
        self._swing_foot_ori_weight = torch.tensor([5000, 5000, 5000])
        """

        #right now using config task weights
        self._com_z_task_weight = 80*torch.ones(self.n_batch)
        self._torso_ori_weight = 80*torch.ones(self.n_batch)
        self._swing_foot_weight = 40*torch.ones(self.n_batch)
        self._stance_foot_weight = 60*torch.ones(self.n_batch)
        self._stance_foot_ori_weight = 60*torch.ones(self.n_batch)
        self._swing_foot_ori_weight = 40*torch.ones(self.n_batch)
    
    """
    self._pos = np.copy(value[0:3, 3])
    self._quat = R.from_matrix(value[0:3, 0:3]).as_quat()
    self._rot = np.copy(value[0:3, 0:3])
    self._iso = np.copy(value)
    """

    def initializeOri(self):
        """ TODO: change when robot changed and have orbit functions
        des_torso_rot = self._robot.get_link_iso(self.torso_id)[:, 0:3, 0:3]

        self._des_torso_rot = util.MakeHorizontalDirX(des_torso_rot)

        self.des_lfoot_iso = self._des_torso_rot.clone().detach()
        self.des_rfoot_iso = self._des_torso_rot.clone().detach()

        self.des_ori_torso = util.rot_to_quat_pytorch(self._des_torso_rot)
        self.des_ori_lfoot = self.des_ori_torso.clone().detach()
        self.des_ori_rfoot = self.des_ori_torso.clone().detach()

        """
        des_torso_rot = self._robot.get_link_iso(self.torso_id)[:, 0:3, 0:3]
        #maybe add make horizontal
        self._des_torso_rot = util.MakeHorizontalDirX(des_torso_rot)

        self.des_lfoot_iso = self._des_torso_rot.clone().detach()
        self.des_rfoot_iso = self._des_torso_rot.clone().detach()


        des_torso_rot = self._des_torso_rot[0, :, :].clone().detach().numpy() #change when orbit rot

        des_ori_torso = util.rot_to_quat(des_torso_rot)
        self.des_ori_torso = torch.from_numpy(des_ori_torso).expand(self.n_batch, -1)
        self.des_ori_lfoot = self.des_ori_torso.clone().detach()
        self.des_ori_rfoot = self.des_ori_torso.clone().detach()





    def setNewOri(self): #performs rotation of com_yaw angle along z axis
        """ TODO: change when new robot and orbit
        des_torso_rot = self._robot.get_link_iso(self.torso_id)[:, 0:3, 0:3]
        self._des_torso_rot = util.MakeHorizontalDirX(des_torso_rot)
        self._des_torso_rot = util.rotationZ(self.des_com_yaw, self._des_torso_rot)

        self.des_lfoot_iso = self._des_torso_rot.clone().detach()
        self.des_rfoot_iso = self._des_torso_rot.clone().detach()
        #TODO: check if we need to normalize quats
        self.des_ori_torso = util.rot_to_quat_pytorch(self._des_torso_rot)
        self.des_ori_lfoot = self.des_ori_torso.clone().detach()
        self.des_ori_rfoot = self.des_ori_torso.clone().detach()        
        """

        des_torso_rot = self._robot.get_link_iso(self.torso_id)[:, 0:3, 0:3]

        self._des_torso_rot = util.MakeHorizontalDirX(des_torso_rot)
        self._des_torso_rot = util.rotationZ(self.des_com_yaw, self._des_torso_rot)

        self.des_lfoot_iso = self._des_torso_rot.clone().detach()
        self.des_rfoot_iso = self._des_torso_rot.clone().detach()

        des_torso_rot = self._des_torso_rot[0, :, :].clone().detach().numpy() #change when orbit rot
        des_ori_torso = util.rot_to_quat(des_torso_rot)    

        self.des_ori_torso = torch.from_numpy(des_ori_torso).expand(self.n_batch, -1)
        self.des_ori_lfoot = self.des_ori_torso.clone().detach()
        self.des_ori_rfoot = self.des_ori_torso.clone().detach()   

        

    

    def generateSwingFtraj(self, start_time, tr_, swfoot_end):
        assert self._stance_leg != None

        self.tr_ = tr_

        self.Swingfoot_end = swfoot_end
        self.swing_start_time = start_time

        #TODO: change to full batched stance leg 
        if (self._stance_leg[0] == 1): # LF is swing foor
            curr_swfoot_iso = self._robot.get_link_iso(self._lfoot_task.target_id)
        else:
            curr_swfoot_iso = self._robot.get_link_iso(self._rfoot_task.target_id)

        curr_swfoot_pos = curr_swfoot_iso[:, 0:3, 3]

        #self.AlipSwing2_curve = interpolation.AlipSwing2(self.Swingfoot_start, self.Swingfoot_end, self.swing_height, self.tr_)
        self.AlipSwing2_curve = interpolation.AlipSwing2(curr_swfoot_pos, self.Swingfoot_end, self.swing_height, self.tr_)

    def updateDesired(self, curr_time):
        assert self._stance_leg != None

        t = curr_time - self.swing_start_time

        self.des_sw_foot_pos = self.AlipSwing2_curve.evaluate(t)
        self.des_sw_foot_vel = self.AlipSwing2_curve.evaluate_first_derivative(t)
        self.des_sw_foot_acc =self.AlipSwing2_curve.evaluate_second_derivative(t)

        #TODO: change to batched stance leg
        if (self._stance_leg[0] == 1):  #update left
            self._lfoot_task.update_desired(self.des_sw_foot_pos, self.des_sw_foot_vel, self.des_sw_foot_acc)
            self.updateCurrentPos(self._rfoot_task)

            self._lfoot_task.w_hierarchy = self._swing_foot_weight
            self._rfoot_task.w_hierarchy = self._stance_foot_weight

            self._lfoot_ori_task.w_hierarchy = self._swing_foot_ori_weight
            self._rfoot_ori_task.w_hierarchy = self._stance_foot_ori_weight
        else:
            self._rfoot_task.update_desired(self.des_sw_foot_pos, self.des_sw_foot_vel, self.des_sw_foot_acc)
            self.updateCurrentPos(self._lfoot_task)

            self._rfoot_task.w_hierarchy = self._swing_foot_weight
            self._lfoot_task.w_hierarchy = self._stance_foot_weight

            self._rfoot_ori_task.w_hierarchy = self._swing_foot_ori_weight
            self._lfoot_ori_task.w_hierarchy = self._stance_foot_ori_weight

        self._torso_ori_task.w_hierarchy = self._torso_ori_weight

        self._torso_ori_task.update_desired(self.des_ori_torso, 
                                           torch.zeros(self.n_batch, 3),
                                           torch.zeros(self.n_batch,3))
        self._rfoot_ori_task.update_desired(self.des_ori_rfoot, 
                                           torch.zeros(self.n_batch, 3),
                                           torch.zeros(self.n_batch,3))
        self._lfoot_ori_task.update_desired(self.des_ori_lfoot, 
                                           torch.zeros(self.n_batch, 3),
                                           torch.zeros(self.n_batch,3))
        """COM z is not implemented.
            Full COM task is implemented
        self._com_z_task.w_hierarchy = self.com_z_task_weight
        self._com_z_task.update_desired(self.refzH*torch.ones(self.n_batch,1),
                                       torch.zeros(self.n_batch, 1),
                                       torch.zeros(self.n_batch,1))
        """
        #TODO: change when robot
        com_pos = self._robot.get_com_pos().clone()
        com_vel = self._robot.get_com_lin_vel().clone()

        com_pos[:, 2] = self.refzH*torch.ones(self.n_batch, dtype=torch.float32)
        com_vel[:, 2] = torch.zeros(self.n_batch)

        self._com_task.w_hierarchy = self._com_z_task_weight
        self._com_task.update_desired(com_pos, com_vel, torch.zeros(self.n_batch, 3))




    def updateCurrentPos(self, task):
        des_iso = self._robot.get_link_iso(task.target_id)
        des_pos = des_iso[:, 0:3, 3]
        task.update_desired(des_pos,
                            torch.zeros(self.n_batch, 3),
                            torch.zeros(self.n_batch,3))
    
    @property
    def des_torso_rot(self):
        return self._des_torso_rot

    @property
    def stance_leg(self):
        return self._stance_leg

    @stance_leg.setter
    def stance_leg(self, val):
        self._stance_leg = val
        

