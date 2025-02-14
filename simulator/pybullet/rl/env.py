import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import torch


import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil

import cv2

from config.draco3_alip_config import SimConfig
from config.draco3_alip_config import AlipParams

from pnc_pytorch.draco3_pnc.draco3_interface import Draco3Interface
from util import pybullet_util_rl

GRIPPER_JOINTS = [
    "left_ezgripper_knuckle_palm_L1_1", "left_ezgripper_knuckle_L1_L2_1",
    "left_ezgripper_knuckle_palm_L1_2", "left_ezgripper_knuckle_L1_L2_2",
    "right_ezgripper_knuckle_palm_L1_1", "right_ezgripper_knuckle_L1_L2_1",
    "right_ezgripper_knuckle_palm_L1_2", "right_ezgripper_knuckle_L1_L2_2"
]

def set_initial_config(robot, joint_id, client):
    # Upperbody
    client.resetJointState(robot, joint_id["l_shoulder_aa"], np.pi / 6, 0.)
    client.resetJointState(robot, joint_id["l_elbow_fe"], -np.pi / 2, 0.)
    client.resetJointState(robot, joint_id["r_shoulder_aa"], -np.pi / 6, 0.)
    client.resetJointState(robot, joint_id["r_elbow_fe"], -np.pi / 2, 0.)

    # Lowerbody
    hip_yaw_angle = 5
    client.resetJointState(robot, joint_id["l_hip_aa"], np.radians(hip_yaw_angle),
                      0.)
    client.resetJointState(robot, joint_id["l_hip_fe"], -np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["l_knee_fe_jp"], np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["l_knee_fe_jd"], np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["l_ankle_fe"], -np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["l_ankle_ie"],
                      np.radians(-hip_yaw_angle), 0.)

    client.resetJointState(robot, joint_id["r_hip_aa"], np.radians(-hip_yaw_angle),
                      0.)
    client.resetJointState(robot, joint_id["r_hip_fe"], -np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["r_knee_fe_jp"], np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["r_knee_fe_jd"], np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["r_ankle_fe"], -np.pi / 4, 0.)
    client.resetJointState(robot, joint_id["r_ankle_ie"], np.radians(hip_yaw_angle),
                      0.)
    
def dict_to_numpy(obs_dict):
    obs = []
    for k,v in obs_dict.items():
        if isinstance(v,dict):
            for k2,v2 in v.items():
                obs.append(v2)
        elif isinstance(v,np.ndarray) or isinstance(v,list) or isinstance(v,tuple):
            for i in range(len(v)):
                obs.append(v[i])
        else:
            obs.append(v)
    return np.array(obs)
    
class DracoEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    def __init__(self, render: bool = False) -> None:
        self.render = render
        if self.render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            self.client.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        
        self.action_space = gym.spaces.Box(  #maximum and minumni value
            low = np.array([-0.1, -0.1, -0.1]),
            high = np.array([0.1, 0.1, 0.1]),
            dtype = np.float64
        )
        
        """self.observation_space = gym.spaces.Dict(
            {
                "base_com_pos" : gym.spaces.Box(
                    low=np.array([-100.0,-100.0,-100.0]), high=np.array([100.0,100.0,100.0]),dtype=np.float32),
                "base_com_quat" : gym.spaces.Box(
                    low=np.array([-2,-2,-2,-2]), high=np.array([2,2,2,2]),dtype=np.float32),
                "base_com_lin_vel" : gym.spaces.Box(
                    low=np.array([-100.0,-100.0,-100.0]), high=np.array([100.0,100.0,100.0]),dtype=np.float32),
                "base_com_ang_vel" : gym.spaces.Box(
                    low=np.array([-100.0,-100.0,-100.0]), high=np.array([100.0,100.0,100.0]),dtype=np.float32),
                "base_joint_pos" : gym.spaces.Box(
                    low=np.array([-100.0,-100.0,-100.0]), high=np.array([100.0,100.0,100.0]),dtype=np.float32),
                "base_joint_quat" : gym.spaces.Box(
                    low=np.array([-7.0,-7.0,-7.0,-7.0]), high=np.array([7.0,7.0,7.0,7.0]),dtype=np.float32),
                "base_joint_lin_vel" : gym.spaces.Box(
                    low=np.array([-100.0,-100.0,-100.0]), high=np.array([100.0,100.0,100.0]),dtype=np.float32),
                "base_joint_ang_vel" : gym.spaces.Box(
                    low=np.array([-100.0,-100.0,-100.0]), high=np.array([100.0,100.0,100.0]),dtype=np.float32),
                "joint_pos" : gym.spaces.Dict(
                    {
                        "neck_pitch" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_shoulder_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_shoulder_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_shoulder_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_elbow_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_wrist_ps" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_wrist_pitch" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_shoulder_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_shoulder_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_shoulder_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_elbow_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_wrist_ps" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_wrist_pitch" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_hip_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_hip_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_hip_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_knee_fe_jp" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_knee_fe_jd" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_ankle_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_ankle_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_hip_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_hip_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_hip_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_knee_fe_jp" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_knee_fe_jd" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_ankle_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_ankle_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32), 
                    }
                ),
                "joint_vel" : gym.spaces.Dict(
                    {
                        "neck_pitch" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_shoulder_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_shoulder_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_shoulder_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_elbow_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_wrist_ps" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_wrist_pitch" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_shoulder_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_shoulder_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_shoulder_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_elbow_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_wrist_ps" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_wrist_pitch" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_hip_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_hip_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_hip_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_knee_fe_jp" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_knee_fe_jd" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_ankle_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "l_ankle_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_hip_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_hip_aa" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_hip_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_knee_fe_jp" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_knee_fe_jd" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_ankle_fe" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32),
                        "r_ankle_ie" : gym.spaces.Box(low=-7.0,high=7.0,dtype=np.float32), 
                    }
                ),
                "b_rf_contact" : gym.spaces.MultiBinary(1),
                "b_lf_contact" : gym.spaces.MultiBinary(1),
            }
        )"""

        self.observation_space = gym.spaces.Box(  #observation space
            low = np.array([-100]*94),
            high = np.array([100]*94),
            dtype = np.float64
        )

        #reward terms
        self._w_roll_pitch = 0.05
        self._w_com_height = 0.05

        self._w_desired_Lxy = 0.05
        self._w_desired_yaw = 0.05
        self._w_excessive_fp = 0.05
        self._w_excessive_angle = 0.05
        self._w_termination = -4
        self._w_alive_bonus = 0.5

        self._Lx_main = 0.5*AlipParams.WIDTH*AlipParams.MASS*math.sqrt(AlipParams.G/AlipParams.ZH) \
                        *AlipParams.ZH*math.tanh(math.sqrt(AlipParams.G/AlipParams.ZH)*AlipParams.TS/2)
        
        #initialise old_wbc_obs for reward
        self._old_wbc_obs = torch.zeros(AlipParams.N_BATCH, 18)
        self._new_wbc_obs = torch.zeros(AlipParams.N_BATCH, 18)

    def reset(self, seed: int = 0):  #creates env
        # Environment Setup
        self.client.resetSimulation()
        if (self.render):
            self.client.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=120,
                cameraPitch=-30,
                cameraTargetPosition=[1, 0.5, 1.0])
        self.client.setGravity(0, 0, -9.8)
        self.client.setPhysicsEngineParameter(
            fixedTimeStep=SimConfig.CONTROLLER_DT, numSubSteps=SimConfig.N_SUBSTEP)
        if SimConfig.VIDEO_RECORD:
            video_dir = 'video/draco3_pnc'
            if os.path.exists(video_dir):
                shutil.rmtree(video_dir)
            os.makedirs(video_dir)

        # Create Robot, Ground
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.robot = self.client.loadURDF(cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
                        SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

        self.client.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
        self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        nq, nv, na,self.joint_id,self.link_id, self.pos_basejoint_to_basecom, self.rot_basejoint_to_basecom = pybullet_util_rl.get_robot_config(
            self.robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
            SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO,  client = self.client)

        # Add Gear constraint
        c = self.client.createConstraint(
            self.robot,
           self.link_id['l_knee_fe_lp'],
            self.robot,
           self.link_id['l_knee_fe_ld'],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0])
        self.client.changeConstraint(c, gearRatio=-1, maxForce=500, erp=10)

        c = self.client.createConstraint(
            self.robot,
           self.link_id['r_knee_fe_lp'],
            self.robot,
           self.link_id['r_knee_fe_ld'],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0])
        self.client.changeConstraint(c, gearRatio=-1, maxForce=500, erp=10)

        # Initial Config
        set_initial_config(self.robot, self.joint_id, self.client)

        # Link Damping
        pybullet_util_rl.set_link_damping(self.robot, self.link_id.values(), 0., 0., client=self.client)

        # Joint Friction
        pybullet_util_rl.set_joint_friction(self.robot,self.joint_id, 0., client=self.client)
        gripper_attached_joint_id = OrderedDict()
        gripper_attached_joint_id["l_wrist_pitch"] = self.joint_id["l_wrist_pitch"]
        gripper_attached_joint_id["r_wrist_pitch"] = self.joint_id["r_wrist_pitch"]
        pybullet_util_rl.set_joint_friction(self.robot, gripper_attached_joint_id, 0.1, client=self.client)

        nominal_sensor_data = pybullet_util_rl.get_sensor_data(
        self.robot,self.joint_id,self.link_id, self.pos_basejoint_to_basecom,
        self.rot_basejoint_to_basecom, client=self.client)

        self.gripper_command = dict()
        for gripper_joint in GRIPPER_JOINTS:
            self.gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
                gripper_joint]
            
        self.obs = copy.deepcopy(nominal_sensor_data)

        for gripper_joint in GRIPPER_JOINTS:
            del self.obs['joint_pos'][gripper_joint]
            del self.obs['joint_vel'][gripper_joint]

        rf_height = pybullet_util_rl.get_link_iso(self.robot,
                                              self.link_id['r_foot_contact'], client=self.client)[2, 3]
        lf_height = pybullet_util_rl.get_link_iso(self.robot,
                                              self.link_id['l_foot_contact'], client=self.client)[2, 3]
        self.obs['b_rf_contact'] = True if rf_height <= 0.01 else False
        self.obs['b_lf_contact'] = True if lf_height <= 0.01 else False

        self.interface = Draco3Interface()

        info = {
            "gripper_command" : self.gripper_command,
            "interface" : self.interface,
            }
        obs_numpy  = dict_to_numpy(self.obs)
        obs_numpy = np.concatenate((obs_numpy, np.zeros(12)))
        
        return obs_numpy, info
    
    def step(self, action):
        #residual, self.gripper_command = action[0], action[1]

        # TODO remove printing

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).unsqueeze(0)
        step_flag = [False]
        wbc_obs = None
        while not step_flag[0]:
            self.obs, self.policy_obs = self._get_observation(wbc_obs)

            command, step_flag, wbc_obs = self.interface.get_command((self.obs, action)) # TODO pass in residual
            self._set_motor_command(command)
            self.client.stepSimulation()
            if self.render: time.sleep(SimConfig.CONTROLLER_DT)
            done = self._compute_termination(wbc_obs)
            if done: break


        self.obs, self.policy_obs = self._get_observation(wbc_obs)


            

        reward = self._compute_reward(wbc_obs, action, done)
        info = {
            "gripper_command" : self.gripper_command,
            }


        return self.policy_obs, reward, done, done, info # need terminated AND truncated

    def close(self):
        self.client.disconnect()
        self.client = None

    def _set_motor_command(self, command) -> None:
        # Exclude Knee Distal Joints Command
        del command['joint_trq']['l_knee_fe_jd']
        del command['joint_trq']['r_knee_fe_jd']

        # Apply Command
        pybullet_util_rl.set_motor_trq(self.robot,self.joint_id, command['joint_trq'], client=self.client)
        pybullet_util_rl.set_motor_pos(self.robot,self.joint_id, self.gripper_command, client=self.client)

    def _get_observation(self, wbc_obs = None) -> dict:

         # Get SensorData
        obs = pybullet_util_rl.get_sensor_data(self.robot,self.joint_id,self.link_id,
                                                    self.pos_basejoint_to_basecom,
                                                    self.rot_basejoint_to_basecom, client=self.client)
        
        for gripper_joint in GRIPPER_JOINTS:
            del obs['joint_pos'][gripper_joint]
            del obs['joint_vel'][gripper_joint]

        rf_height = pybullet_util_rl.get_link_iso(self.robot,
                                              self.link_id['r_foot_contact'], client=self.client)[2, 3]
        lf_height = pybullet_util_rl.get_link_iso(self.robot,
                                              self.link_id['l_foot_contact'], client=self.client)[2, 3]
        obs['b_rf_contact'] = True if rf_height <= 0.01 else False
        obs['b_lf_contact'] = True if lf_height <= 0.01 else False

        if wbc_obs is not None:
            obs_numpy  = dict_to_numpy(self.obs)
            wbc_np = wbc_obs[0].numpy()
            obs_numpy  = dict_to_numpy(self.obs)
            obs_numpy[0:3] -= wbc_np[12:15]
            policy_obs = np.concatenate((obs_numpy, wbc_np[0:12]))
        else:
            policy_obs = None
        return obs, policy_obs
    
    def _compute_termination(self, _wbc_obs):
        #TODO: add more termination
        condition = torch.any((_wbc_obs[:, 8] < 0.5) | (_wbc_obs[:, 8] > 0.8))  #0.69
        return condition
        #return False

    def _compute_reward(self, wbc_obs, action, done):
        self._old_wbc_obs = self._new_wbc_obs
        self._new_wbc_obs = wbc_obs
        self._rl_action = action

        reward = self._w_alive_bonus
        reward += self.reward_tracking_com_L()
        reward += self.reward_tracking_yaw()
        reward += self.reward_com_height()
        reward += self.reward_roll_pitch()
        reward += self.penalise_excessive_fp()
        reward += self.penalise_excessive_yaw()
        if done: reward -= self._w_termination

        return reward.squeeze().item()

    def reward_tracking_com_L(self):
        Lx = torch.zeros(AlipParams.N_BATCH, 2)
        Lx[:, 0] = self._new_wbc_obs[:, 0]*self._Lx_main 
        #in the code 1 corresponds to current stance foot right
        # -1 to current stance foot left 
        # new obs -1 --> ended policy for left foot --> we are at the desired state for end of right stance
        error = Lx + self._old_wbc_obs[:, 1:3] - self._new_wbc_obs[:, 9:11]  #desired Lx,y - observedLx,y at the end of the step
        error = torch.sum(torch.square(error), dim = 1)
        error *= self._w_desired_Lxy
        return torch.exp(-error)
    
    def reward_tracking_yaw(self):
        error = self._new_wbc_obs[:, 16] - self._old_wbc_obs[:, 16] - self._old_wbc_obs[:, 3]
        error = torch.square(error)
        error *= self._w_desired_yaw

        return torch.exp(-error)

    def reward_com_height(self):
        error = self._new_wbc_obs[:, 8] - AlipParams.ZH
        error = torch.square(error)
        error *= self._w_com_height
        return torch.exp(-error)

    def reward_roll_pitch(self):
        error = torch.sum(torch.square(self._new_wbc_obs[:, 14:16]), dim = 1)
        error *= self._w_roll_pitch 
        return torch.exp(-error)
    
    def penalise_excessive_fp(self):
        error = torch.sum(torch.square(self._rl_action[:, 0:2]), dim = 1)
        error *= self._w_excessive_fp
        return torch.exp(-error)
    
    def penalise_excessive_yaw(self): 
        error = torch.square(self._rl_action[:, 2])
        error *= self._w_excessive_angle
        return torch.exp(-error)


if __name__ == "__main__":
    env = DracoEnv(True)

    from stable_baselines3.common.env_checker import check_env
    #check_env(env)

    obs, info = env.reset()
    interface = info["interface"]

    while True:
        action = torch.ones(AlipParams.N_BATCH,3)
        obs, reward, done, trunc, info = env.step(action)
        if done: 
            obs,info = env.reset()


