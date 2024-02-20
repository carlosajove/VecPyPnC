import abc
import torch


class InternalConstraint(abc.ABC):
    """
    WBC Internal Constraint
    Usage:
        update_internal_constraint
    """
    def __init__(self, robot, dim, n_batch):
        self._robot = robot
        self._dim = dim
        self.n_batch = n_batch
        self._jacobian = torch.zeros(self.n_batch, self._dim, self._robot.n_q_dot, dtype = torch.double)
        self._jacobian_dot_q_dot = torch.zeros(self.n_batch, self._dim, dtype=torch.double)

    @property
    def jacobian(self):
        return self._jacobian

    @property
    def jacobian_dot_q_dot(self):
        return self._jacobian_dot_q_dot

    def update_internal_constraint(self):
        self._update_jacobian()

    @abc.abstractmethod
    def _update_jacobian(self):
        pass
