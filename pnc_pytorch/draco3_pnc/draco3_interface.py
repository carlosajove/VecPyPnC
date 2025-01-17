import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
import copy

import pybullet as p

from config.draco3_alip_config import PnCConfig
from config.draco3_alip_config import AlipParams
from pnc_pytorch.interface import Interface
from pnc_pytorch.draco3_pnc.draco3_interrupt_logic import Draco3InterruptLogic
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider
from pnc_pytorch.draco3_pnc.draco3_state_estimator import Draco3StateEstimator
from pnc_pytorch.draco3_pnc.draco3_control_architecture import Draco3ControlArchitecture
from pnc_pytorch.data_saver import DataSaver


class Draco3Interface(Interface):
    def __init__(self):
        super(Draco3Interface, self).__init__()
        self._n_batch = AlipParams.N_BATCH
        if PnCConfig.DYN_LIB == "dart":
            from pnc_pytorch.robot_system.dart_robot_system import DartRobotSystem
            self._robot = DartRobotSystem(
                "/home/junhyeok/Repository/PnC/RobotModel/draco/draco_rel_path.urdf",
                False, False)
        elif PnCConfig.DYN_LIB == "pinocchio":
            from pnc_pytorch.robot_system.pinocchio_robot_system import PinocchioRobotSystem
            self._robot = PinocchioRobotSystem(self._n_batch, 
                cwd + "/robot_model/draco3/draco3.urdf",
                cwd + "/robot_model/draco3", False, PnCConfig.PRINT_ROBOT_INFO)
        else:
            raise ValueError("wrong dynamics library")

        self._sp = Draco3StateProvider(self._robot, self._n_batch)
        self._sp.reset(self._robot, self._n_batch)
        self._se = Draco3StateEstimator(self._robot, self._n_batch)
        self._control_architecture = Draco3ControlArchitecture(self._robot, self._n_batch)
        self._interrupt_logic = Draco3InterruptLogic(
            self._control_architecture)
        if PnCConfig.SAVE_DATA:
            self._data_saver = DataSaver()
            self._data_saver.add('joint_pos_limit',
                                 self._robot.joint_pos_limit)
            self._data_saver.add('joint_vel_limit',
                                 self._robot.joint_vel_limit)
            self._data_saver.add('joint_trq_limit',
                                 self._robot.joint_trq_limit)

    def get_command(self, input_command, verbose = False):
        if PnCConfig.SAVE_DATA:
            self._data_saver.add('time', self._running_time)
            self._data_saver.add('phase', self._control_architecture.state)

        sensor_data = input_command[0]
        rl_action = input_command[1]
        # Update State Estimator
        if self._count == 0 and verbose:
            print("=" * 80)
            print("Initialize")
            print("=" * 80)
        if self._count == 0:
            self._se.initialize(sensor_data)
        self._se.update(sensor_data)
        self._se.inertia_to_com_torso_coor()

        # Process Interrupt Logic
        #self._interrupt_logic.process_interrupts()

        # Compute Cmd
        command, trigger, rl_obs = self._control_architecture.get_command(rl_action)
        #print("interface", command)

        if PnCConfig.SAVE_DATA and (self._count % PnCConfig.SAVE_FREQ == 0):
            self._data_saver.add('joint_pos', self._robot.joint_positions)
            self._data_saver.add('joint_vel', self._robot.joint_velocities)
            self._data_saver.advance()

        # Increase time variables
        self._count += 1
        self._running_time += PnCConfig.CONTROLLER_DT
        self._sp.curr_time = self._running_time
        self._sp.prev_state = self._control_architecture.prev_state
        self._sp.state = self._control_architecture.state

        res = (command, trigger, rl_obs)
        #assert False
        return copy.deepcopy(res)

    @property
    def interrupt_logic(self):
        return self._interrupt_logic
