from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import torch
import math

from util import liegroup
from util import orbit_util
from util import util


def smooth_changing(ini, end, dur, curr_time):
    ret = ini + (end - ini) * 0.5 * (1 - np.cos(curr_time / dur * np.pi))
    if curr_time > dur:
        ret = end

    return ret


def smooth_changing_vel(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (np.pi / dur) * np.sin(curr_time / dur * np.pi)
    if curr_time > dur:
        ret = 0.

    return ret


def smooth_changing_acc(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (np.pi / dur) * (
        np.pi / dur) * np.cos(curr_time / dur * np.pi)
    if curr_time > dur:
        ret = 0.
    return ret

def smooth_changing_pytorch(ini, end, dur, curr_time):
    ret = ini + (end - ini) * 0.5 * (1 - torch.cos(curr_time / dur * math.pi))
    ret = torch.where(curr_time > dur, end, ret)
    return ret


def smooth_changing_vel_pytorch(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (math.pi / dur) * torch.sin(curr_time / dur * math.pi)
    ret = torch.where(curr_time > dur, end, ret)
    return ret


def smooth_changing_acc_pytorch(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (math.pi / dur) * (
        math.pi / dur) * torch.cos(curr_time / dur * torch.pi)
    ret = torch.where(curr_time > dur, end, ret)
    return ret


def iso_interpolate(T1, T2, alpha):
    p1 = T1[0:3, 3]
    R1 = T1[0:3, 0:3]
    p2 = T2[0:3, 3]
    R2 = T2[0:3, 0:3]

    slerp = Slerp([0, 1], R.from_matrix([R1, R2]))

    p_ret = alpha * (p1 + p2)
    R_ret = slerp(alpha).as_matrix()

    return liegroup.RpToTrans(R_ret, p_ret)


class HermiteCurve(object):
    def __init__(self, start_pos, start_vel, end_pos, end_vel):
        self._p1 = start_pos
        self._v1 = start_vel
        self._p2 = end_pos
        self._v2 = end_vel

    def evaluate(self, s_in):
        s = np.clip(s_in, 0., 1.)
        return self._p1 * (2 * s**3 - 3 * s**2 + 1) + self._p2 * (
            -2 * s**3 + 3 * s**2) + self._v1 * (
                s**3 - 2 * s**2 + s) + self._v2 * (s**3 - s**2)

    def evaluate_first_derivative(self, s_in):
        s = np.clip(s_in, 0., 1.)

        return self._p1 * (6 * s**2 - 6 * s) + self._p2 * (
            -6 * s**2 + 6 * s) + self._v1 * (
                3 * s**2 - 4 * s + 1) + self._v2 * (3 * s**2 - 2 * s)

    def evaluate_second_derivative(self, s_in):
        s = np.clip(s_in, 0., 1.)

        return self._p1 * (12 * s - 6) + self._p2 * (
            -12 * s + 6) + self._v1 * (6 * s - 4) + self._v2 * (6 * s - 2)


class HermiteCurveVec(object):
    def __init__(self, start_pos, start_vel, end_pos, end_vel):
        self._p1 = np.copy(start_pos)
        self._v1 = np.copy(start_vel)
        self._p2 = np.copy(end_pos)
        self._v2 = np.copy(end_vel)
        self._dim = start_pos.shape[0]

        self._curves = []
        for i in range(self._dim):
            self._curves.append(
                HermiteCurve(start_pos[i], start_vel[i], end_pos[i],
                             end_vel[i]))

    def evaluate(self, s_in):
        return np.array([c.evaluate(s_in) for c in self._curves])

    def evaluate_first_derivative(self, s_in):
        return np.array(
            [c.evaluate_first_derivative(s_in) for c in self._curves])

    def evaluate_second_derivative(self, s_in):
        return np.array(
            [c.evaluate_second_derivative(s_in) for c in self._curves])


class HermiteCurveQuat(object):
    def __init__(self, quat_start, ang_vel_start, quat_end, ang_vel_end):
        self._qa = R.from_quat(quat_start)
        self._omega_a = np.copy(ang_vel_start)
        self._qb = R.from_quat(quat_end)
        self._omega_b = np.copy(ang_vel_end)

        # Initialize Data Structures
        self._q0 = R.from_quat(quat_start)

        if np.linalg.norm(ang_vel_start) < 1e-6:
            self._q1 = R.from_quat(quat_start) * R.from_quat([0., 0., 0., 1.])
        else:
            self._q1 = R.from_quat(quat_start) * R.from_rotvec(
                (np.linalg.norm(ang_vel_start) / 3.0) *
                (ang_vel_start / np.linalg.norm(ang_vel_start)))

        if np.linalg.norm(ang_vel_end) < 1e-6:
            self._q2 = R.from_quat(quat_end) * R.from_quat([0., 0., 0., 1.])
        else:
            self._q2 = R.from_quat(quat_end) * R.from_rotvec(
                (np.linalg.norm(ang_vel_end) / 3.0) *
                (ang_vel_end / np.linalg.norm(ang_vel_end)))

        self._q3 = R.from_quat(quat_end)

        self._omega_1aa = self._q1 * self._q0.inv()
        self._omega_2aa = self._q2 * self._q1.inv()
        self._omega_3aa = self._q3 * self._q2.inv()

        self._omega_1 = self._omega_1aa.as_rotvec()
        self._omega_2 = self._omega_2aa.as_rotvec()
        self._omega_3 = self._omega_3aa.as_rotvec()

    def _compute_basis(self, s_in):
        s = np.clip(s_in, 0., 1.)

        self._b1 = 1 - (1 - s)**3
        self._b2 = 3 * s**2 - 2 * s**3
        self._b3 = s**3
        self._bdot1 = 3 * (1 - s)**2
        self._bdot2 = 6 * s - 6 * s**2
        self._bdot3 = 3 * s**2
        self._bddot1 = -6 * (1 - s)
        self._bddot2 = 6 - 12 * s
        self._bddot3 = 6 * s

    def evaluate(self, s_in):
        s = np.clip(s_in, 0., 1.)
        self._compute_basis(s)

        if np.linalg.norm(self._omega_1) > 1e-5:
            qtmp1 = R.from_rotvec(
                (np.linalg.norm(self._omega_1) * self._b1) *
                (self._omega_1 / np.linalg.norm(self._omega_1)))
        else:
            qtmp1 = R.from_quat([0., 0., 0., 1.])
        if np.linalg.norm(self._omega_2) > 1e-5:
            qtmp2 = R.from_rotvec(
                (np.linalg.norm(self._omega_2) * self._b2) *
                (self._omega_2 / np.linalg.norm(self._omega_2)))
        else:
            qtmp2 = R.from_quat([0., 0., 0., 1.])
        if np.linalg.norm(self._omega_3) > 1e-5:
            qtmp3 = R.from_rotvec(
                (np.linalg.norm(self._omega_3) * self._b3) *
                (self._omega_3 / np.linalg.norm(self._omega_3)))
        else:
            qtmp3 = R.from_quat([0., 0., 0., 1.])

        return (qtmp3 * qtmp2 * qtmp1 * self._q0).as_quat()

    def evaluate_ang_vel(self, s_in):
        s = np.clip(s_in, 0., 1.)
        self._compute_basis(s)

        return self._omega_1 * self._bdot1 + self._omega_2 * self._bdot2 + self._omega_3 * self._bdot3

    def evaluate_ang_acc(self, s_in):
        s = np.clip(s_in, 0., 1.)
        self._compute_basis(s)

        return self._omega_1 * self._bddot1 + self._omega_2 * self._bddot2 + self._omega_3 * self._bddot3




class HermiteCurveQuat_torch_test(object):
    #https://dl.acm.org/doi/pdf/10.1145/218380.218486
    #quaternion formalism will be (w, x,, y, z)

    def __init__(self, n_batch):
        self._n_batch = n_batch
        self._q0 = torch.zeros(n_batch, 4)
        self._w1 = torch.zeros(n_batch, 4)
        self._w2 = torch.zeros(n_batch, 3)
        self._w3 = torch.zeros(n_batch, 3)

        self._qa = torch.zeros(n_batch, 4)
        self._qb = torch.zeros(n_batch, 4)
        self._wa = torch.zeros(n_batch, 3)
        self._wb = torch.zeros(n_batch, 3)
    
    def setParams(self, id, q_a, q_b, w_a, w_b): #id must be a list of indexes
        self._qa[id] = q_a
        self._qb[id] = q_b
        self._wa[id] = w_a
        self._wb[id] = w_b

        self._q0 = self._qa
        self._w1 = self._wa/3.
        self._w3 = self._wb/3.
        self._w2 = util.log_quat_map(orbit_util.quat_mul(
                                     orbit_util.quat_mul(orbit_util.quat_inv(util.exp_quat_map(self._w1)),
                                                         orbit_util.quat_inv(self._qa)),
                                     orbit_util.quat_mul(self._qb,
                                                         util.exp_quat_map(self._w3))))

    def _compute_basis(self, s_in):
        s = torch.clamp(s_in, 0., 1.).unsqueeze(1)

        self._b1 = 1 - (1 - s)**3
        self._b2 = 3 * s**2 - 2 * s**3
        self._b3 = s**3
        self._bdot1 = 3 * (1 - s)**2
        self._bdot2 = 6 * s - 6 * s**2
        self._bdot3 = 3 * s**2
        self._bddot1 = -6 * (1 - s)
        self._bddot2 = 6 - 12 * s
        self._bddot3 = 6 * s

    def evaluate(self, s_in):
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)
        res = orbit_util.quat_mul(orbit_util.quat_mul(orbit_util.quat_mul(
                                        self._q0, 
                                        util.exp_quat_map(self._w1 * self._b1)),
                                        util.exp_quat_map(self._w2 * self._b2)),
                                        util.exp_quat_map(self._w3 * self._b3))
        return res

    def evaluate_ang_vel(self, s_in):
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)

        return self._w1 * self._bdot1 + self._w2 * self._bdot2 + self._w3* self._bdot3
    
    def evaluate_ang_acc(self, s_in):
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)

        return self._w1 * self._bddot1 + self._w2 * self._bddot2 + self._w3* self._bddot3



    
"""TODO: maybe finish implementation
         need to find efficient way of doing the if 
class AlipSwing(object):
    def __init__(self, start_pos, end_pos, mid_z_pos, duration):
        self._start_pos = start_pos.clone().detach()
        self._end_pos = end_pos.clone().detach()
        self._mid_z_pos = mid_z_pos.clone().detach()
        self._duration = duration.clone().detach()
        self.n_batch = self._start_pos.shape[0]

        self.first_z = torchHermiteCurve(self._start_pos[:, 2], torch.ones(self.n_batch), 
                                         self._mid_z_pos, torch.zeros(self.n_batch), self._duration/2)
        self.second_z = torchHermiteCurve(self._mid_z_pos, torch.zeros(self.n_batch), 
                                         self._end_pos[:, 2], torch.ones(self.n_batch), self._duration/2)
    """
"""
    t is a batched tensor, different t in each sim, because different contact
    """
"""
    def evaluate(self, t): 
        s = t/self._duration

        x = 0.5*((1+torch.cos(math.pi*s))*self._start_pos[:, 0] + (1-torch.cos(math.pi*s))*self._end_pos[:, 0]);
        y = 0.5*((1+torch.cos(math.pi*s))*self._start_pos[:, 1] + (1-torch.cos(math.pi*s))*self._end_pos[:, 1]);
        
        if (t > self._duration):
            z = self.first_z.evaluate(t - self._duration/2)
        else:


        return torch.stach((x,y,z), dim = 1)
    
    def evaluate_first_derivative(self, t):
"""

class AlipSwing2(object): # input is batched
    def __init__(self, start_pos, end_pos, mid_z_pos, duration):
        self._start_pos = start_pos.clone().detach()
        self._end_pos = end_pos.clone().detach()
        self._mid_z_pos = mid_z_pos.clone().detach()
        self._duration = duration.clone().detach()
        self.n_batch = self._start_pos.shape[0]

        self.z_curve = QuadraticLagrangePol(start_pos[: , 2], torch.zeros(self.n_batch), 
                                            self._mid_z_pos , self._duration/2,
                                            self._end_pos[:, 2], self._duration)
    
    def evaluate(self, t):
        _t = torch.min(t, self._duration)  

        s = _t/self._duration
        #if s > 1 return end pos

        #x = 0.5*((1+torch.cos(math.pi*s))*self._start_pos[:, 0] + (1-torch.cos(math.pi*s))*self._end_pos[:, 0])
        #y = 0.5*((1+torch.cos(math.pi*s))*self._start_pos[:, 1] + (1-torch.cos(math.pi*s))*self._end_pos[:, 1])

        x = 0.5*(self._start_pos[:, 0] + self._end_pos[:, 0] + torch.cos(math.pi*s)*(self._start_pos[:, 0] - self._end_pos[:, 0]))
        y = 0.5*(self._start_pos[:, 1] + self._end_pos[:, 1] + torch.cos(math.pi*s)*(self._start_pos[:, 1] - self._end_pos[:, 1]))
        z = self.z_curve.evaluate(_t)

        return torch.stack((x,y,z), dim = 1)
    
    def evaluate_first_derivative(self, t):
        _t = torch.min(t, self._duration)  
        s = _t/self._duration
        #if s > 1 return end pos

        x = 0.5*math.pi*torch.sin(math.pi*s)/self._duration*(self._end_pos[:, 0] - self._start_pos[:, 0])
        y = 0.5*math.pi*torch.sin(math.pi*s)/self._duration*(self._end_pos[:, 1] - self._start_pos[:, 1])
        
        z = self.z_curve.evaluate_first_derivative(_t)

        return torch.stack((x, y, z), dim = 1)
    
    def evaluate_second_derivative(self, t):
        _t = torch.min(t, self._duration)  
        s = _t/self._duration
        #if s > 1 return end pos
        s = torch.min(s, torch.tensor(1.0))

        x = 0.5*math.pi*math.pi*torch.cos(math.pi*s)/self._duration/self._duration * (self._end_pos[:, 0] - self._start_pos[:, 0])
        y = 0.5*math.pi*math.pi*torch.cos(math.pi*s)/self._duration/self._duration * (self._end_pos[:, 1] - self._start_pos[:, 1])
        z = self.z_curve.evaluate_second_derivative(_t)

        return torch.stack((x, y, z), dim = 1)




class QuadraticLagrangePol(object): #sizes are torhc.tensor([n_batch])
    def __init__(self, z0, t0, z1, t1, z2, t2):
        self._z0 = z0.clone()
        self._z1 = z1.clone()
        self._z2 = z2.clone()
        self._t0 = t0.clone()
        self._t1 = t1.clone()
        self._t2 = t2.clone()

        self._c0 = self._z0/((self._t0 - self._t1)*(self._t0 - self._t2))
        self._c1 = self._z1/((self._t1 - self._t0)*(self._t1 - self._t2))
        self._c2 = self._z2/((self._t2 - self._t0)*(self._t2 - self._t1))

    def evaluate(self, t):
        output = (t - self._t1)*(t - self._t2)*self._c0 + \
                 (t - self._t0)*(t - self._t2)*self._c1 + \
                 (t - self._t0)*(t - self._t1)*self._c2
        return output
        
    def evaluate_first_derivative(self, t):
        output = self._c0*(2*t - self._t2 - self._t1) + \
                 self._c1*(2*t - self._t0 - self._t2) + \
                 self._c2*(2*t - self._t0 - self._t1) 
        return output

    def evaluate_second_derivative(self, t):
        return 2*(self._c0 + self._c1 + self._c2)

#TODO: erase and do hermite
class LinearQuatCurve():
    def __init__(self, start_quat, end_quat, duration):
        self._st_quat = start_quat
        self._end_quat = end_quat
        self._duration = duration
        self._diff = (self._end_quat - self._st_quat)/self._duration.unsqueeze(1)

    def evaluate(self, t):
        return self._diff * t.unsqueeze(1) + self._st_quat

    def evaluate_first_derivative(self, t):
        #return self._diff
        return torch.zeros(self._st_quat.shape[0], 3)

    def evaluate_second_derivative(self, t):
        return torch.zeros(self._st_quat.shape[0], 3)