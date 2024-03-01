import torch 
from util import util
from util import interpolation
from util import orbit_util

from pnc_pytorch.data_saver import DataSaver
from config.draco3_alip_config import WBCConfig, AlipParams

#everithing has the shape [n_batch , normal_shape]
class ALIPtrajectoryManager(object):
    def __init__(self, _n_batch, com_task, torso_ori_task, lfoot_task, lfoot_ori_task,
                rfoot_task, rfoot_ori_task, robot):
        self._n_batch = _n_batch
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
        self.Swingfoot_start = torch.zeros(self._n_batch, 3, dtype = torch.double)
        self.Swingfoot_end = torch.zeros(self._n_batch, 3, dtype = torch.double)
        self.stleg_pos = torch.zeros(self._n_batch, 3, dtype = torch.double)

        self.stleg_pos_torso_ori = torch.zeros(self._n_batch, 3, dtype = torch.double)  #frame of ref is torso ori 

        self.des_swfoot_pos = torch.zeros(self._n_batch, 3, dtype = torch.double)
        self.des_sw_foot_vel = torch.zeros(self._n_batch, 3, dtype = torch.double)
        self.des_swfoot_acc = torch.zeros(self._n_batch, 3, dtype = torch.double)

        self.des_ori_lfoot = torch.tensor([0, 0, 0, 1], dtype = torch.double).unsqueeze(0).repeat(self._n_batch, 1, 1)
        self.des_ori_rfoot = torch.tensor([0, 0, 0, 1], dtype = torch.double).unsqueeze(0).repeat(self._n_batch, 1, 1)
        self.des_ori_torso = torch.tensor([0, 0, 0, 1], dtype = torch.double).unsqueeze(0).repeat(self._n_batch, 1, 1)

        self._des_torso_rot = torch.eye(3, dtype = torch.double).unsqueeze(0).repeat(self._n_batch, 1, 1)
        self.des_lfoot_rot = torch.eye(3, dtype = torch.double).unsqueeze(0).repeat(self._n_batch, 1, 1)
        self.des_rfoot_rot = torch.eye(3, dtype = torch.double).unsqueeze(0).repeat(self._n_batch, 1, 1)
        self._des_com_yaw  = torch.zeros(self._n_batch, dtype = torch.double) #rotation per step

        #indata variables mpc frame of refrence (stance foot frame of ref)
        self._stance_leg = AlipParams.INITIAL_STANCE_LEG * torch.ones(self._n_batch)
        self._stance_leg[-1] = self._stance_leg[-1]*-1
        self.Ts = AlipParams.TS

        #time vars
        self.swing_start_time = torch.zeros(self._n_batch, dtype = torch.double)
        #self._initial_swing_foot_pos = torch.zeros(self._n_batch, 3, dtype = torch.double)
        #parameters:#set from parameters in ctrl arch
        self.swing_height = AlipParams.SWING_HEIGHT*torch.ones(self._n_batch, dtype = torch.double)
        self.refzH = AlipParams.ZH


        #right now using config task weights
        self._com_z_task_weight      = WBCConfig.W_COM * torch.ones(self._n_batch, dtype = torch.double)
        self._torso_ori_weight       = WBCConfig.W_COM * torch.ones(self._n_batch, dtype = torch.double)
        self._swing_foot_weight      = WBCConfig.W_SWING_FOOT * torch.ones(self._n_batch, dtype = torch.double)
        self._stance_foot_weight     = WBCConfig.W_CONTACT_FOOT * torch.ones(self._n_batch, dtype = torch.double)
        self._stance_foot_ori_weight = WBCConfig.W_CONTACT_FOOT * torch.ones(self._n_batch, dtype = torch.double)
        self._swing_foot_ori_weight  = WBCConfig.W_SWING_FOOT * torch.ones(self._n_batch, dtype = torch.double)

        #Create curves classes
        self.hermite_quat_torso = interpolation.HermiteCurveQuat_torch_test(self._n_batch)
        self.hermite_quat_swfoot = interpolation.HermiteCurveQuat_torch_test(self._n_batch)
        self.AlipSwing2_curve = interpolation.AlipSwing2(self._n_batch)



    def initializeOri(self):
        #""" TODO: change when robot changed and have orbit functions
        des_torso_rot = self._robot.get_link_iso(self.torso_id)[:, 0:3, 0:3]
        self._des_torso_rot = util.MakeHorizontalDirX(des_torso_rot)

        self.des_lfoot_iso = self._des_torso_rot.clone().detach()
        self.des_rfoot_iso = self._des_torso_rot.clone().detach()

        self.des_ori_torso = orbit_util.convert_quat(orbit_util.quat_from_matrix(self._des_torso_rot))
        self.des_ori_lfoot = self.des_ori_torso.clone().detach()
        self.des_ori_rfoot = self.des_ori_torso.clone().detach()



    def setNewOri(self, ids): #performs rotation of com_yaw angle along z axis
        torso_rot = self._robot.get_link_iso(self.torso_id)[ids, 0:3, 0:3]
        des_torso_rot = util.MakeHorizontalDirX(torso_rot)
        self._des_torso_rot[ids] = util.rotationZ(self._des_com_yaw[ids], des_torso_rot)


        self.des_lfoot_iso[ids] = self._des_torso_rot[ids].clone().detach()
        self.des_rfoot_iso[ids] = self._des_torso_rot[ids].clone().detach()
        

        self.des_ori_torso[ids] = orbit_util.convert_quat(orbit_util.quat_from_matrix(self._des_torso_rot[ids]))
        self.des_ori_lfoot[ids] = self.des_ori_torso[ids].clone().detach()
        self.des_ori_rfoot[ids] = self.des_ori_torso[ids].clone().detach()        
        



        
    #create AlipSwing
    #don't create a new interpolation, but update the class, with list of batch
    def generateSwingFtraj(self, start_time, tr_, swfoot_end, ids):
        assert len(ids) > 0
        assert self._stance_leg != None

        self.swing_start_time[ids] = start_time

        curr_swfoot_iso = torch.where(self._stance_leg[ids].unsqueeze(1).unsqueeze(1) == 1, self._robot.get_link_iso(self._lfoot_task.target_id)[ids],
                                                                                            self._robot.get_link_iso(self._rfoot_task.target_id)[ids])
        swfoot_rot = torch.where(self._stance_leg[ids].unsqueeze(1).unsqueeze(1) == 1, self._robot.get_link_iso(self._lfoot_task.target_id)[ids, 0:3, 0:3],
                                                                                       self._robot.get_link_iso(self._rfoot_task.target_id)[ids, 0:3, 0:3])
        

        curr_swfoot_pos = curr_swfoot_iso[ids, 0:3, 3]

        self.AlipSwing2_curve.setParams(ids, curr_swfoot_pos, swfoot_end, self.swing_height[ids], tr_)
        
        #ori
        torso_rot = self._robot.get_link_iso(self.torso_id)[ids, 0:3, 0:3]
        ori_torso_quat = orbit_util.quat_from_matrix(torso_rot)
        swfoot_quat = orbit_util.quat_from_matrix(swfoot_rot)

        qbswing = orbit_util.convert_quat(self.des_ori_lfoot[ids], to = "wxyz")
        qbstorso = orbit_util.quat_from_matrix(self._des_torso_rot[ids])
        self.hermite_quat_torso.setParams(ids, ori_torso_quat, qbstorso, torch.zeros(len(ids), 3, dtype = torch.double),
                                                                         torch.zeros(len(ids), 3, dtype = torch.double), tr_)
        self.hermite_quat_swfoot.setParams(ids, swfoot_quat, qbswing, torch.zeros(len(ids), 3, dtype = torch.double),
                                                                      torch.zeros(len(ids), 3, dtype = torch.double), tr_)



    def updateDesired(self, curr_time, ids):
        assert len(ids) > 0
        assert self._stance_leg != None

        #Get Desired states from trajectories
        t = curr_time - self.swing_start_time

        des_torso_quat = orbit_util.convert_quat(self.hermite_quat_torso.evaluate(t))[ids]
        des_torso_quat_v  = self.hermite_quat_torso.evaluate_ang_vel(t)[ids]
        des_torso_quat_a = self.hermite_quat_torso.evaluate_ang_acc(t)[ids]

        des_swfoot_quat = orbit_util.convert_quat(self.hermite_quat_swfoot.evaluate(t))[ids]
        des_swfoot_quat_v  = self.hermite_quat_swfoot.evaluate_ang_vel(t)[ids]
        des_swfoot_quat_a = self.hermite_quat_swfoot.evaluate_ang_acc(t)[ids]

        self.des_sw_foot_pos = self.AlipSwing2_curve.evaluate(t)[ids]
        self.des_sw_foot_vel = self.AlipSwing2_curve.evaluate_first_derivative(t)[ids]
        self.des_sw_foot_acc =self.AlipSwing2_curve.evaluate_second_derivative(t)[ids]
        
        ################################
        # UPDATE THE ORIENTATION TASKS #
        ################################
        self._torso_ori_task.update_desired(des_torso_quat, 
                                            des_torso_quat_v,
                                            des_torso_quat_a, ids)

        rfoot_quat = orbit_util.convert_quat(orbit_util.quat_from_matrix(self._robot.get_link_iso(self.rfoot_id)[ids, 0:3, 0:3]))
        lfoot_quat = orbit_util.convert_quat(orbit_util.quat_from_matrix(self._robot.get_link_iso(self.lfoot_id)[ids, 0:3, 0:3]))
        des_rfoot_quat = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, rfoot_quat, des_swfoot_quat[ids])
        des_rfoot_ang_vel = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, torch.zeros(self._n_batch, 3, dtype = torch.double), des_swfoot_quat_v[ids])
        des_rfoot_ang_acc = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, torch.zeros(self._n_batch, 3, dtype = torch.double), des_swfoot_quat_a[ids])
        des_lfoot_quat = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, des_swfoot_quat[ids], lfoot_quat)
        des_lfoot_ang_vel = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, des_swfoot_quat_v[ids], torch.zeros(self._n_batch, 3, dtype = torch.double))
        des_lfoot_ang_acc = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, des_swfoot_quat_a[ids], torch.zeros(self._n_batch, 3, dtype = torch.double))

        self._rfoot_ori_task.update_desired(des_rfoot_quat, des_rfoot_ang_vel, des_rfoot_ang_acc, ids)
        self._lfoot_ori_task.update_desired(des_lfoot_quat, des_lfoot_ang_vel, des_lfoot_ang_acc, ids)

        #############################
        # UPDATE THE POSITION TASKS #
        #############################
        rfootpos = self._robot.get_link_iso(self.rfoot_id)[ids, 0:3, 3]
        rfootpos[ids,2] = torch.zeros(len(ids), dtype = torch.double)
        lfootpos = self._robot.get_link_iso(self.lfoot_id)[ids, 0:3, 3]
        lfootpos[:,2] = torch.zeros(len(ids), dtype = torch.double)

        des_rfoot_pos = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, rfootpos, self.des_sw_foot_pos[ids])
        des_rfoot_vel = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, torch.zeros(self._n_batch, 3, dtype = torch.double), self.des_sw_foot_vel[ids])
        des_rfoot_acc = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, torch.zeros(self._n_batch, 3, dtype = torch.double), self.des_sw_foot_acc[ids])
        des_lfoot_pos = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, self.des_sw_foot_pos[ids], lfootpos)
        des_lfoot_vel = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, self.des_sw_foot_vel[ids], torch.zeros(self._n_batch, 3, dtype = torch.double))
        des_lfoot_acc = torch.where(self._stance_leg[ids].unsqueeze(1) == 1, self.des_sw_foot_acc[ids], torch.zeros(self._n_batch, 3, dtype = torch.double))
        self._rfoot_task.update_desired(des_rfoot_pos, des_rfoot_vel, des_rfoot_acc, ids)
        self._lfoot_task.update_desired(des_lfoot_pos, des_lfoot_vel, des_lfoot_acc, ids)

        ####################
        # SET TASK WEIGHTS #
        ####################
        rfoot_task_hierarchy = torch.where(self._stance_leg == 1, self._stance_foot_weight, self._swing_foot_weight)
        rfoot_ori_task_hierarchy = torch.where(self._stance_leg == 1, self._stance_foot_ori_weight, self._swing_foot_ori_weight)
        lfoot_task_hierarchy = torch.where(self._stance_leg == 1, self._swing_foot_weight, self._stance_foot_weight)
        lfoot_ori_task_hierarchy = torch.where(self._stance_leg == 1, self._swing_foot_ori_weight, self._stance_foot_ori_weight)
        self._rfoot_task.w_hierarchy = rfoot_task_hierarchy
        self._lfoot_task.w_hierarchy = lfoot_task_hierarchy
        self._lfoot_ori_task.w_hierarchy = lfoot_ori_task_hierarchy
        self._rfoot_ori_task.w_hierarchy = rfoot_ori_task_hierarchy
        self._torso_ori_task.w_hierarchy = self._torso_ori_weight

        ############
        # COM TASK #
        ############
        com_pos = self._robot.get_com_pos().clone()[ids]
        com_vel = self._robot.get_com_lin_vel().clone()[ids]
        #com_pos[:, 1] = torch.zeros(self._n_batch)
        #com_vel[:, 1] = torch.zeros(self._n_batch)
        com_pos[:, 2] = self.refzH*torch.ones(len(ids), dtype=torch.double)
        com_vel[:, 2] = torch.zeros(len(ids), dtype = torch.double)

        self._com_task.w_hierarchy = self._com_z_task_weight
        self._com_task.update_desired(com_pos, com_vel, torch.zeros(len(ids), 3, dtype = torch.double), ids)

        





    def updateCurrentPos(self, task): #stance foot pos, z = 0 hardcoded
        des_iso = self._robot.get_link_iso(task.target_id)
        des_pos = des_iso[:, 0:3, 3]
        des_pos[:,2] = torch.zeros(self._n_batch)
        task.update_desired(des_pos,
                            torch.zeros(self._n_batch, 3, dtype = torch.double),
                            torch.zeros(self._n_batch, 3, dtype = torch.double))
    
    def updateCurrentOri(self, task): 
        des_iso = self._robot.get_link_iso(task.target_id)
        des_rot = des_iso[:, 0:3, 0:3]
        des_rot = orbit_util.convert_quat(orbit_util.quat_from_matrix(des_rot))
        task.update_desired(des_rot,
                            torch.zeros(self._n_batch, 3, dtype = torch.double),
                            torch.zeros(self._n_batch, 3, dtype = torch.double))


    def use_both_current(self):
        self.updateCurrentPos(self._lfoot_task)
        self.updateCurrentOri(self._lfoot_ori_task)
        self.updateCurrentPos(self._rfoot_task)
        self.updateCurrentOri(self._rfoot_ori_task)
       
    
    @property
    def des_torso_rot(self):
        return self._des_torso_rot

    @property
    def stance_leg(self):
        return self._stance_leg

    @property
    def des_com_yaw(self):
        return self._des_com_yaw

    #setter
    def stance_leg(self, val, ids = None):
        if ids is None:
            self._stance_leg = val 
        else:
            self._stance_leg[ids] = val
    #setter
    def des_com_yaw(self, val, ids = None):
        if ids is None:
            self._des_com_yaw = val  
        else:    
            self._des_com_yaw[ids] = val

