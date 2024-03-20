import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

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

from config.draco3_config import SimConfig
from pnc.draco3_pnc.draco3_interface import Draco3Interface
from util import pybullet_util

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

        self.action_space = gym.spaces.Box(
            low = np.array([-2, -2, -2]),
            high = np.array([2, 2, 2]),
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

        self.observation_space = gym.spaces.Box(
            low = np.array([-100]*82),
            high = np.array([100]*82),
            dtype = np.float64
        )
    
    def reset(self, seed: int = 0):
        # Environment Setup
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
        nq, nv, na,self.joint_id,self.link_id, self.pos_basejoint_to_basecom, self.rot_basejoint_to_basecom = pybullet_util.get_robot_config(
            self.robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
            SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

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
        pybullet_util.set_link_damping(self.robot, self.link_id.values(), 0., 0.)

        # Joint Friction
        pybullet_util.set_joint_friction(self.robot,self.joint_id, 0.)
        gripper_attached_joint_id = OrderedDict()
        gripper_attached_joint_id["l_wrist_pitch"] = self.joint_id["l_wrist_pitch"]
        gripper_attached_joint_id["r_wrist_pitch"] = self.joint_id["r_wrist_pitch"]
        pybullet_util.set_joint_friction(self.robot, gripper_attached_joint_id, 0.1)

        nominal_sensor_data = pybullet_util.get_sensor_data(
        self.robot,self.joint_id,self.link_id, self.pos_basejoint_to_basecom,
        self.rot_basejoint_to_basecom)

        self.gripper_command = dict()
        for gripper_joint in GRIPPER_JOINTS:
            self.gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
                gripper_joint]
            
        self.obs = copy.deepcopy(nominal_sensor_data)

        for gripper_joint in GRIPPER_JOINTS:
            del self.obs['joint_pos'][gripper_joint]
            del self.obs['joint_vel'][gripper_joint]

        rf_height = pybullet_util.get_link_iso(self.robot,
                                              self.link_id['r_foot_contact'])[2, 3]
        lf_height = pybullet_util.get_link_iso(self.robot,
                                              self.link_id['l_foot_contact'])[2, 3]
        self.obs['b_rf_contact'] = True if rf_height <= 0.01 else False
        self.obs['b_lf_contact'] = True if lf_height <= 0.01 else False

        self.interface = Draco3Interface()

        info = {
            "gripper_command" : self.gripper_command,
            "interface" : self.interface,
            }
        obs_numpy  = dict_to_numpy(self.obs)
            
        return obs_numpy, info
    
    def step(self, action):
        #residual, self.gripper_command = action[0], action[1]

        # TODO remove printing

        step_flag = False
        while not step_flag:
            command, step_flag, policy_obs = self.interface.get_command(self.obs) # TODO pass in residual
        
        self._set_motor_command(command)

        self.client.stepSimulation()
        if self.render: time.sleep(SimConfig.CONTROLLER_DT)

        self.obs = self._get_observation()
        reward = self._compute_reward()
        done = self._compute_termination()
        info = {
            "gripper_command" : self.gripper_command,
            }
        obs_numpy  = dict_to_numpy(self.obs)

        return obs_numpy, reward, done, done, info # need terminated AND truncated

    def close(self):
        self.client.disconnect()
        self.client = None

    def _set_motor_command(self, command) -> None:
        # Exclude Knee Distal Joints Command
        del command['joint_pos']['l_knee_fe_jd']
        del command['joint_pos']['r_knee_fe_jd']
        del command['joint_vel']['l_knee_fe_jd']
        del command['joint_vel']['r_knee_fe_jd']
        del command['joint_trq']['l_knee_fe_jd']
        del command['joint_trq']['r_knee_fe_jd']

        # Apply Command
        pybullet_util.set_motor_trq(self.robot,self.joint_id, command['joint_trq'])
        pybullet_util.set_motor_pos(self.robot,self.joint_id, self.gripper_command)

    def _get_observation(self) -> dict:

         # Get SensorData
        obs = pybullet_util.get_sensor_data(self.robot,self.joint_id,self.link_id,
                                                    self.pos_basejoint_to_basecom,
                                                    self.rot_basejoint_to_basecom)
        
        for gripper_joint in GRIPPER_JOINTS:
            del obs['joint_pos'][gripper_joint]
            del obs['joint_vel'][gripper_joint]

        rf_height = pybullet_util.get_link_iso(self.robot,
                                              self.link_id['r_foot_contact'])[2, 3]
        lf_height = pybullet_util.get_link_iso(self.robot,
                                              self.link_id['l_foot_contact'])[2, 3]
        obs['b_rf_contact'] = True if rf_height <= 0.01 else False
        obs['b_lf_contact'] = True if lf_height <= 0.01 else False

        return obs
    
    def _compute_reward(self):
        # TODO
        # impliment reward terms
        return 0

    def _compute_termination(self):
        # TODO
        # impliment termination conditions
        return False

if __name__ == "__main__":
    env = DracoEnv(True)

    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    obs, info = env.reset()
    interface = info["interface"]

    while True:
        obs, reward, done, trunc, info = env.step(None)