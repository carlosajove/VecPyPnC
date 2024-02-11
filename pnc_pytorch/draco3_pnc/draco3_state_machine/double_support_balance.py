import numpy as np

from config.draco3_config import WalkingState
from pnc_pytorch.state_machine import StateMachine
from pnc_pytorch.draco3_pnc.draco3_state_provider import Draco3StateProvider


class DoubleSupportBalance(StateMachine):
    def __init__(self, batch, id, tm, robot):
        super(DoubleSupportBalance, self).__init__(id, robot)
        self._n_batch = batch
        self._trajectory_managers = tm
        self._sp = Draco3StateProvider()
        self._start_time = 0.
        self._walking_trigger = False

    @property
    def walking_trigger(self):
        return self._walking_trigger

    @walking_trigger.setter
    def walking_trigger(self, val):
        self._walking_trigger = val

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Foot Task
        self._trajectory_managers.use_both_current()

    def first_visit(self):
        print("[WalkingState] BALANCE")
        self._walking_trigger = False
        self._start_time = self._sp.curr_time

    def last_visit(self):
        pass

    def end_of_state(self):
        if (self._walking_trigger) :
            return True
        return False

    def get_next_state(self):
        return WalkingState.ALIP
