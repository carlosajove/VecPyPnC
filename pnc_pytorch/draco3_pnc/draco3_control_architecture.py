import torch 

from config.draco3_alip_config import WBCConfig, WalkingState
from pnc_pytorch.control_architecture import ControlArchitecture
from pnc_pytorch.wbc.manager.upper_body_trajectory_manager import UpperBodyTrajectoryManager
from pnc_pytorch.draco3_pnc.draco3_tci_container import Draco3TCIContainer
from pnc_pytorch.draco3_pnc.draco3_controller import Draco3Controller
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider


from pnc_pytorch.planner.locomotion.alip_mpc import ALIPtorch_mpc
from pnc_pytorch.wbc.manager.ALIP_trajectory_manager import ALIPtrajectoryManager
from pnc_pytorch.draco3_pnc.draco3_state_machine.alip_locomotion import AlipLocomotion


class Draco3ControlArchitecture(ControlArchitecture):
    def __init__(self, robot, batch):
        super(Draco3ControlArchitecture, self).__init__(robot)
        self._n_batch = batch
        # ======================================================================
        # Initialize TCIContainer
        # ======================================================================
        self._tci_container = Draco3TCIContainer(robot, self._n_batch)

        # ======================================================================
        # Initialize Controller
        # ======================================================================
        self._draco3_controller = Draco3Controller(self._tci_container, robot, self._n_batch)

        # ======================================================================
        # Initialize Planner
        # ======================================================================
        self._alip_mpc = ALIPtorch_mpc(robot, self._n_batch)

        # ======================================================================
        # Initialize Task Manager
        # ======================================================================
        self._alip_tm = ALIPtrajectoryManager(self._n_batch,
                    self._tci_container.com_task, self._tci_container.torso_ori_task,
                    self._tci_container.lfoot_pos_task, self._tci_container.lfoot_ori_task,
                    self._tci_container.rfoot_pos_task, self._tci_container.rfoot_ori_task,
                    robot)


        self._upper_body_tm = UpperBodyTrajectoryManager(self._n_batch,
            self._tci_container.upper_body_task, robot)

        """
        self._trajectory_managers = {
            "rfoot": self._rfoot_tm,
            "lfoot": self._lfoot_tm,
            "upper_body": self._upper_body_tm,
            "floating_base": self._floating_base_tm,
            "dcm": self._dcm_tm
        }
        """

        # ======================================================================
        # Initialize Hierarchy Manager
        # ======================================================================
       
        # Currently not using

        # ======================================================================
        # Initialize Reaction Force Manager
        # ======================================================================

        # Currently not using

        # ======================================================================
        # Initialize State Machines
        # ======================================================================
        
        self._state_machine[WalkingState.ALIP] = AlipLocomotion(self._n_batch, WalkingState.ALIP, self._alip_tm, 
                                                    self._alip_mpc, self._tci_container, robot)
        
    

        # Set Starting State
        self._state = WalkingState.ALIP
        self._prev_state = WalkingState.ALIP
        self._b_state_first_visit = True

        self._sp = Draco3StateProvider()

        self._alip_iter = 0

    def get_command(self):

        if self._b_state_first_visit:
            self._state_machine[self._state].first_visit()
            self._b_state_first_visit = False

        # Update State Machine
        if(self._alip_iter >= 0):
            if (self._alip_iter == 0):
                self._state_machine[self._state].new_step()
            self._state_machine[self._state].one_step()

        # Update State Machine Independent Trajectories
        self._upper_body_tm.use_nominal_upper_body_joint_pos(
            self._sp.nominal_joint_pos)
        # Get Whole Body Control Commands
        command = self._draco3_controller.get_command()
        self._alip_iter += 1
        if (self._state_machine[self._state].switchLeg()):
            self._alip_iter = -3

        return command

    @property
    def state_machine(self):
        return self._state_machine
