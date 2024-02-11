import torch 


class ReactionForceManager(object):
    def __init__(self, batch, contact, maximum_rf_z_max):
        self._n_batch = batch
        self._contact = contact
        self._maximum_rf_z_max = maximum_rf_z_max * torch.ones(self._n_batch)
        self._minimum_rf_z_max = 0.001 * torch.ones(self._n_batch)
        self._starting_rf_z_max = contact.rf_z_max 
        self._start_time = torch.zeros(self._n_batch)
        self._duration = torch.zeros(self._n_batch)

    def initialize_ramp_to_min(self, start_time, duration):
        self._start_time = start_time * torch.ones(self._n_batch)
        self._duration = duration  * torch.ones(self._n_batch)
        self._starting_rf_z_max = self._contact.rf_z_max

    def initialize_ramp_to_max(self, start_time, duration):
        self._start_time = start_time * torch.ones(self._n_batch)
        self._duration = duration * torch.ones(self._n_batch)
        self._starting_rf_z_max = self._contact.rf_z_max

    def update_ramp_to_min(self, current_time):
        current_time = current_time * torch.ones(self._n_batch)
        t = torch.clamp(current_time, self._start_time,
                    self._start_time + self._duration)
        self._contact.rf_z_max = (
            self._minimum_rf_z_max - self._starting_rf_z_max
        ) / self._duration * (t - self._start_time) + self._starting_rf_z_max

    def update_ramp_to_max(self, current_time):
        current_time = current_time * torch.ones(self._n_batch)
        t = torch.clamp(current_time, self._start_time,
                    self._start_time + self._duration)
        self._contact.rf_z_max = (
            self._maximum_rf_z_max - self._starting_rf_z_max
        ) / self._duration * (t - self._start_time) + self._starting_rf_z_max

