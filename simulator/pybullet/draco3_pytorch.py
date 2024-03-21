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
import pybullet as p
import numpy as np
import torch

np.set_printoptions(precision=2)

from config.draco3_alip_config import SimConfig
from pnc_pytorch.draco3_pnc.draco3_interface import Draco3Interface
from util import pybullet_util
from util import util
from util import liegroup
from config.draco3_alip_config import AlipParams

gripper_joints = [
    "left_ezgripper_knuckle_palm_L1_1", "left_ezgripper_knuckle_L1_L2_1",
    "left_ezgripper_knuckle_palm_L1_2", "left_ezgripper_knuckle_L1_L2_2",
    "right_ezgripper_knuckle_palm_L1_1", "right_ezgripper_knuckle_L1_L2_1",
    "right_ezgripper_knuckle_palm_L1_2", "right_ezgripper_knuckle_L1_L2_2"
]


def set_initial_config(robot, joint_id):
    # Upperbody
    p.resetJointState(robot, joint_id["l_shoulder_aa"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["l_elbow_fe"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_shoulder_aa"], -np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["r_elbow_fe"], -np.pi / 2, 0.)

    # Lowerbody
    hip_yaw_angle = 5
    p.resetJointState(robot, joint_id["l_hip_aa"], np.radians(hip_yaw_angle),
                      0.)
    p.resetJointState(robot, joint_id["l_hip_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_knee_fe_jp"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_knee_fe_jd"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_ankle_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_ankle_ie"],
                      np.radians(-hip_yaw_angle), 0.)

    p.resetJointState(robot, joint_id["r_hip_aa"], np.radians(-hip_yaw_angle),
                      0.)
    p.resetJointState(robot, joint_id["r_hip_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_knee_fe_jp"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_knee_fe_jd"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_ankle_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_ankle_ie"], np.radians(hip_yaw_angle),
                      0.)


class rl_reward(object):
    def __init__(self):
        self._w_roll_pitch = 0.05
        self._w_desired_Lxy = 0.1
        self._w_desired_yaw = 0.1
        self._w_com_height = 0.05
        self._w_excessive_fp = 0.05
        self._w_excessive_angle = 0.05
        #self._w_termination = -100
        self._w_alive_bonus = 1.

        self._Lx_main = 0.5*AlipParams.WIDTH*AlipParams.MASS*math.sqrt(AlipParams.G/AlipParams.ZH) \
                        *AlipParams.ZH*math.tanh(math.sqrt(AlipParams.G/AlipParams.ZH)*AlipParams.TS/2)
        

    def set_initial_obs(self, obs):
        self._new_obs = obs

    def set_action(self, action):
        self._action = action

    

    def compute_reward(self, new_obs):
        self._rl_obs = self._new_obs
        self._new_obs = new_obs

        reward = self._w_alive_bonus
        reward += self.reward_tracking_com_L()
        reward += self.reward_tracking_yaw()
        reward += self.reward_com_height()
        reward += self.reward_roll_pitch()
        reward += self.penalise_excessive_fp()
        reward += self.penalise_excessive_yaw()
        return reward

    def reward_tracking_com_L(self):
        Lx = torch.zeros(AlipParams.N_BATCH, 3)
        Lx[:, 0] = self._new_obs[:, 0]*self._Lx_main 
        #in the code 1 corresponds to current stance foot right
        # -1 to current stance foot left 
        # new obs -1 --> ended policy for left foot --> we are at the desired state for end of right stance
        error = Lx + self._rl_obs[:, 1:3] - self._new_obs[:, 9:11]  #desired Lx,y - observedLx,y at the end of the step
        error = torch.sum(torch.square(error), dim = 1)
        error *= self._w_desired_Lxy
        return torch.exp(-error)
    
    def reward_tracking_yaw(self):
        error = self._new_obs[:, 16] - self._rl_obs[:, 16] - self._rl_obs[:, 3]
        error = torch.sum(torch.square(error), dim = 1)
        error *= self._w_desired_yaw
        return torch.exp(-error)

    def reward_com_height(self):
        error = self._new_obs[:, 8] - AlipParams.ZH
        error = torch.square(error)
        error *= self._w_com_height
        return torch.exp(-error)

    def reward_roll_pitch(self):
        error = torch.sum(torch.square(self._new_obs[:, 14:16]), dim = 1)
        error *= self._w_excess
        return torch.exp(-error)
    
    def penalise_excessive_fp(self):
        error = torch.sum(torch.square(self._rl_action[:, 0:2]), dim = 1)
        error *= self._w_excessive_fp
        return torch.exp(-error)
    
    def penalise_excessive_yaw(self):
        error = torch.square(self._rl_action[:, 2])
        error *= self._w_excessive_angle
        return torch.exp(-error)

def compute_policy_input(sensor_data, wbc_obs):
    sensor_data["base_com_pos"] -= wbc_obs[:, 11:14]
    #sensor_input = sensor_data_to_tensor() #TODO
    wbc_input = wbc_obs[:, 0:12]
    #torch.cat((sensor_input, wbc_input), dim = 1)



def signal_handler(signal, frame):
    if SimConfig.VIDEO_RECORD:
        pybullet_util.make_video(video_dir, False)
    p.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=120,
        cameraPitch=-30,
        cameraTargetPosition=[1, 0.5, 1.0])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(
        fixedTimeStep=SimConfig.CONTROLLER_DT, numSubSteps=SimConfig.N_SUBSTEP)
    if SimConfig.VIDEO_RECORD:
        video_dir = 'video/draco3_pnc'
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # robot = p.loadURDF(cwd + "/robot_model/draco3/draco3.urdf",
    robot = p.loadURDF(cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
                       SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                       SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    # Add Gear constraint
    c = p.createConstraint(
        robot,
        link_id['l_knee_fe_lp'],
        robot,
        link_id['l_knee_fe_ld'],
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=10)

    c = p.createConstraint(
        robot,
        link_id['r_knee_fe_lp'],
        robot,
        link_id['r_knee_fe_ld'],
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=10)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0.)
    gripper_attached_joint_id = OrderedDict()
    gripper_attached_joint_id["l_wrist_pitch"] = joint_id["l_wrist_pitch"]
    gripper_attached_joint_id["r_wrist_pitch"] = joint_id["r_wrist_pitch"]
    pybullet_util.set_joint_friction(robot, gripper_attached_joint_id, 0.1)

    # Construct Interface
    interface = Draco3Interface()

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0
    jpg_count = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)

    gripper_command = dict()
    for gripper_joint in gripper_joints:
        gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
            gripper_joint]

    rl_trigger = [False]*AlipParams.N_BATCH
    #set first obs to 0
    first_wbc_obs = torch.zeros(AlipParams.N_BATCH, 18)
    rl_action = torch.zeros(AlipParams.N_BATCH, 3)
    #reward.set_initial_obs(first_wbc_obs)

    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        for gripper_joint in gripper_joints:
            del sensor_data['joint_pos'][gripper_joint]
            del sensor_data['joint_vel'][gripper_joint]

        rf_height = pybullet_util.get_link_iso(robot,
                                               link_id['r_foot_contact'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot,
                                               link_id['l_foot_contact'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        # Get Keyboard Event
        keys = p.getKeyboardEvents()
        if pybullet_util.is_key_triggered(keys, '8'):
            interface.interrupt_logic.b_interrupt_button_eight = True
        elif pybullet_util.is_key_triggered(keys, '5'):
            interface.interrupt_logic.b_interrupt_button_five = True
        elif pybullet_util.is_key_triggered(keys, '4'):
            interface.interrupt_logic.b_interrupt_button_four = True
        elif pybullet_util.is_key_triggered(keys, '2'):
            interface.interrupt_logic.b_interrupt_button_two = True
        elif pybullet_util.is_key_triggered(keys, '6'):
            interface.interrupt_logic.b_interrupt_button_six = True
        elif pybullet_util.is_key_triggered(keys, '7'):
            interface.interrupt_logic.b_interrupt_button_seven = True
        elif pybullet_util.is_key_triggered(keys, '9'):
            interface.interrupt_logic.b_interrupt_button_nine = True
        elif pybullet_util.is_key_triggered(keys, '0'):
            interface.interrupt_logic.b_interrupt_button_zero = True
        elif pybullet_util.is_key_triggered(keys, 'c'):
            for k, v in gripper_command.items():
                gripper_command[k] += 1.94 / 3.
        elif pybullet_util.is_key_triggered(keys, 'o'):
            for k, v in gripper_command.items():
                gripper_command[k] -= 1.94 / 3.
        elif pybullet_util.is_key_triggered(keys, 'a'):
            interface.interrupt_logic.b_interrupt_button_a = True

        # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        
        

        if(rl_trigger[0] == True): #compute RL 
            policy_input = compute_policy_input(sensor_data, rl_obs)
            rl_action = torch.zeros(AlipParams.N_BATCH, 3) #X, Y, YAW
            #compute_policy(policy_input)
            #reward.set_action(rl_action)

        input_command = (sensor_data, rl_action)

        alip_command = interface.get_command(copy.deepcopy(input_command))

        command = alip_command[0]

        if(rl_trigger[0] == True):
            rl_trigger = alip_command[1]
            #obs = sensordata + rl_obs --> in sensor data we need to change sensor base position data to foot frame
            rl_obs = alip_command[2]
            #reward.compute_reward(rl_obs) #if it fails and never reaches true no reward will be added
                                           #the reward is computed at the end of the step in which the policy has been applied.
                                           #that is when the foot reaches the position specified by the policy
                                           #Maybe the reward should be computed at the end of the step with the stance leg that is in the policy position
                                           #like in the mpc
                                           #ori can just choose the other Lx as target





        if SimConfig.PRINT_TIME:
            end_time = time.time()
            print("ctrl computation time: ", end_time - start_time)

        # Exclude Knee Distal Joints Command
        """
        del command['joint_pos']['l_knee_fe_jd']
        del command['joint_pos']['r_knee_fe_jd']
        del command['joint_vel']['l_knee_fe_jd']
        del command['joint_vel']['r_knee_fe_jd']
        """
        del command['joint_trq']['l_knee_fe_jd']
        del command['joint_trq']['r_knee_fe_jd']

        # Apply Command
        pybullet_util.set_motor_trq(robot, joint_id, command['joint_trq'])
        pybullet_util.set_motor_pos(robot, joint_id, gripper_command)
        #TODO change 
        # Save Image
        if (SimConfig.VIDEO_RECORD) and (count % SimConfig.RECORD_FREQ == 0):
            frame = pybullet_util.get_camera_image(
                [1., 0.5, 1.], 1.0, 120, -15, 0, 60., 1920, 1080, 0.1, 100.)
            frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
            filename = video_dir + '/step%06d.jpg' % jpg_count
            cv2.imwrite(filename, frame)
            jpg_count += 1

        p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1
