import torch 

from config.draco3_alip_config import WBCConfig, WalkingState, PnCConfig
from pnc_pytorch.control_architecture import ControlArchitecture
from pnc_pytorch.wbc.manager.upper_body_trajectory_manager import UpperBodyTrajectoryManager
from pnc_pytorch.draco3_pnc.draco3_tci_container import Draco3TCIContainer
from pnc_pytorch.draco3_pnc.draco3_controller import Draco3Controller
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider


from pnc_pytorch.planner.locomotion.alip_mpc import ALIPtorch_mpc


from pnc_pytorch.wbc.manager.ALIP_trajectory_manager import ALIPtrajectoryManager
from pnc_pytorch.wbc.manager.floating_base_trajectory_manager import FloatingBaseTrajectoryManager

from pnc_pytorch.draco3_pnc.draco3_state_machine.alip_locomotion import AlipLocomotion
from pnc_pytorch.draco3_pnc.draco3_state_machine.double_support_balance import DoubleSupportBalance
from pnc_pytorch.draco3_pnc.draco3_state_machine.double_support_stand import DoubleSupportStand
from pnc_pytorch.wbc.manager.reaction_force_manager import ReactionForceManager

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
        self._alip_mpc = ALIPtorch_mpc(robot, PnCConfig.SAVE_DATA)
        # ======================================================================
        # Initialize Task Manager
        # ======================================================================
        self._alip_tm = ALIPtrajectoryManager(self._n_batch,
                    self._tci_container.com_task, self._tci_container.torso_ori_task,
                    self._tci_container.lfoot_pos_task, self._tci_container.lfoot_ori_task,
                    self._tci_container.rfoot_pos_task, self._tci_container.rfoot_ori_task,
                    robot)

        self._floating_base_tm = FloatingBaseTrajectoryManager(self._n_batch,
                    self._tci_container.com_task, self._tci_container.torso_ori_task,
                    robot)

        self._upper_body_tm = UpperBodyTrajectoryManager(self._n_batch,
            self._tci_container.upper_body_task, robot)

        
        self._trajectory_managers = {
            "alip_tm": self._alip_tm,
            "upper_body": self._upper_body_tm,
            "floating_base": self._floating_base_tm,
        }
        

        # ======================================================================
        # Initialize Hierarchy Manager
        # ======================================================================
       
        # Currently not using

        # ======================================================================
        # Initialize Reaction Force Manager
        # ======================================================================
        #ONLY USING IN STAND UP
        self._rfoot_fm = ReactionForceManager(self._n_batch,
            self._tci_container.rfoot_contact, WBCConfig.RF_Z_MAX)

        self._lfoot_fm = ReactionForceManager(self._n_batch,
            self._tci_container.lfoot_contact, WBCConfig.RF_Z_MAX)

        self._reaction_force_managers = {
            "rfoot": self._rfoot_fm,
            "lfoot": self._lfoot_fm
        }

        # ======================================================================
        # Initialize State Machines
        # ======================================================================
        
        self._state_machine[WalkingState.ALIP] = AlipLocomotion(self._n_batch, WalkingState.ALIP, self._alip_tm, 
                                                    self._alip_mpc, self._tci_container, robot, PnCConfig.SAVE_DATA)
        
        self._state_machine[WalkingState.STAND] = DoubleSupportStand(self._n_batch,
            WalkingState.STAND, self._trajectory_managers, self._reaction_force_managers, robot)
        self._state_machine[
            WalkingState.STAND].end_time = 0.2 
        self._state_machine[
            WalkingState.STAND].rf_z_max_time = 0.1 * torch.ones(self._n_batch)
        self._state_machine[
            WalkingState.STAND].com_height_des = 0.69 * torch.ones(self._n_batch)

        self._state_machine[WalkingState.BALANCE] = DoubleSupportBalance(self._n_batch,
            WalkingState.BALANCE, self._alip_tm, robot)
        
        # Set Starting State
        self._state = WalkingState.STAND
        self._prev_state = WalkingState.STAND
        self._b_state_first_visit = True

        self._sp = Draco3StateProvider()

        self._alip_iter = 0
        self._new_step_list = 3*torch.ones(self._n_batch) # each get_command substracts 1
                                                          # switch_leg set ids to 3 --> tunable parameter
                                                          # new step is computed for ids == 0
                                                          # one step for ids <= 0
                                                        



    def get_command(self):
        #ASSUMES ALL THE SIMULATIONS START WITH ALIP AT THE SAME TIME
        #THERE ARE NO ERRORS BEFORE WALKING STATE ALIP
        if self._b_state_first_visit:
            self._state_machine[self._state].first_visit()
            self._b_state_first_visit = False
        if(self._state == WalkingState.ALIP):
            # Update State Machine
            self._new_step_list -= 1
            print(self._new_step_list)
            ids_new_step = torch.nonzero(self._new_step_list == 0).squeeze().tolist()
            ids_one_step = torch.nonzero(self._new_step_list <= 0).squeeze().tolist()
            if (len(ids_new_step) > 0):
                self._state_machine[self._state].new_step(ids_new_step)
            if (len(ids_one_step) > 0):
                self._state_machine[self._state].one_step(ids_one_step)

            # Update State Machine Independent Trajectories
            self._upper_body_tm.use_nominal_upper_body_joint_pos(
                self._sp.nominal_joint_pos)
            # Get Whole Body Control Commands
            command = self._draco3_controller.get_command()
            self._new_step_list = self._state_machine[self._state].switchLeg(self._new_step_list)
            #self._state_machine[self._state]
        
        else: 
             # Update State Machine
            self._state_machine[self._state].one_step()
            # Update State Machine Independent Trajectories
            self._upper_body_tm.use_nominal_upper_body_joint_pos(
                self._sp.nominal_joint_pos)
            # Get Whole Body Control Commands
            command = self._draco3_controller.get_command()

            if self._state_machine[self._state].end_of_state():
                print("END", "%"*80)
                self._state_machine[self._state].last_visit()
                self._prev_state = self._state
                self._state = self._state_machine[self._state].get_next_state()
                self._b_state_first_visit = True
        return command

    @property
    def state_machine(self):
        return self._state_machine
