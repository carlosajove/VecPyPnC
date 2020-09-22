import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np

from pnc.wbc.contact_spec import Contact


class PointContact(Contact):
    def __init__(self, robot, link_id, mu):
        super(PointContact, self).__init__(robot, 3)

        self._link_id = link_id
        self._max_fz = 500.
        self._mu = mu

    def _update_jacobian(self):
        jt_temp = self._robot.get_link_jac(self._link_id)
        self._jacobian = jt_temp[self._dim_contact:, :]
        self._jacobian_dot_q_dot = np.dot(
            self._robot.get_link_jacobian_dot(
                self._link_id)[self._dim_contact:, :], self._robot.get_qdot())

    def _update_cone_constraint(self):
        rot = self._robot.get_link_iso(self._link_id)[0:3, 0:3].transpose()
        self._uf = np.zeros((6, self._dim_contact))
        self._uf[0, 2] = 1.

        self._uf[1, 0] = 1.
        self._uf[1, 2] = self._mu
        self._uf[2, 0] = -1.
        self._uf[2, 2] = self._mu

        self._uf[3, 1] = 1.
        self._uf[3, 2] = self._mu
        self._uf[4, 1] = -1.
        self._uf[4, 2] = self._mu

        self._uf[5, 2] = -1.

        self._uf = np.dot(self._uf, rot)

        self._ieq_vec = np.zeors(6)
        self._ieq_vec[5] = -self._max_fz


class SurfaceContact(Contact):
    def __init__(self, robot, link_id, x, y, mu):
        super(SurfaceContact, self).__init__(robot, 6)

        self._link_id = link_id
        self._max_fz = 1500.
        self._x = x
        self._y = y
        self._mu = mu

    def _update_jacobian(self):
        self._jacobian = self._robot.get_link_jac(self._link_id)
        self._jacobian_dot_q_dot = np.dot(
            self._robot.get_jacobian_dot(self._link_id), self.get_qdot())

    def _update_cone_constraint(self):
        self._uf = np.zeros((16 + 2, self._dim_contact))

        u = self._get_u(self._x, self._y, self._mu)
        rot = self._robot.get_link_iso(self._link_id)[0:3, 0:3]
        rot_foot = np.zeros((6, 6))
        rot_foot[0:3, 0:3] = rot.transpose()
        rot_foot[3:6, 3:6] = rot.transpose()

        self._uf = np.dot(u, rot_foot)

        self._ieq_vec = np.zeors(16 + 2)
        self._ieq_vec[17] = -self._max_fz

    def _get_u(self, x, y, mu):
        u = np.zeors((16 + 2, 6))

        u[0, 5] = 1.

        u[1, 3] = 1.
        u[1, 5] = mu
        u[2, 3] = -1.
        u[2, 5] = mu

        u[3, 4] = 1.
        u[3, 5] = mu
        u[4, 4] = -1.
        u[4, 5] = mu

        u[5, 0] = 1.
        u[5, 5] = y
        u[6, 0] = -1.
        u[6, 5] = y

        u[7, 1] = 1.
        u[7, 5] = x
        u[8, 1] = -1.
        u[8, 5] = x

        ##tau
        u[9, 0] = -mu
        u[9, 1] = -mu
        u[9, 2] = 1.
        u[9, 3] = y
        u[9, 4] = x
        u[9, 5] = (x + y) * mu

        u[10, 0] = -mu
        u[10, 1] = mu
        u[10, 2] = 1.
        u[10, 3] = y
        u[10, 4] = -x
        u[10, 5] = (x + y) * mu

        u[11, 0] = mu
        u[11, 1] = -mu
        u[11, 2] = 1.
        u[11, 3] = -y
        u[11, 4] = x
        u[11, 5] = (x + y) * mu

        u[12, 0] = mu
        u[12, 1] = mu
        u[12, 2] = 1.
        u[12, 3] = -y
        u[12, 4] = -x
        u[12, 5] = (x + y) * mu

        u[13, 0] = -mu
        u[13, 1] = -mu
        u[13, 2] = -1.
        u[13, 3] = -y
        u[13, 4] = -x
        u[13, 5] = (x + y) * mu

        u[14, 0] = -mu
        u[14, 1] = mu
        u[14, 2] = -1.
        u[14, 3] = -y
        u[14, 4] = x
        u[14, 5] = (x + y) * mu

        u[15, 0] = mu
        u[15, 1] = -mu
        u[15, 2] = -1.
        u[15, 3] = y
        u[15, 4] = -x
        u[15, 5] = (x + y) * mu

        u[16, 0] = mu
        u[16, 1] = mu
        u[16, 2] = -1.
        u[16, 3] = y
        u[16, 4] = x
        u[16, 5] = (x + y) * mu

        u[17, 5] = -1.

    return u
