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
    if (ini.dim() > 1):
        curr_time = curr_time.unsqueeze(1)

    ret = ini + (end - ini) * 0.5 * (1 - torch.cos(curr_time / dur * math.pi))
    ret = torch.where(curr_time > dur, end, ret)
    return ret


def smooth_changing_vel_pytorch(ini, end, dur, curr_time):
    if (ini.dim() > 1):
        curr_time = curr_time.unsqueeze(1)
    ret = (end - ini) * 0.5 * (math.pi / dur) * torch.sin(curr_time / dur * math.pi)
    ret = torch.where(curr_time > dur, end, ret)
    return ret


def smooth_changing_acc_pytorch(ini, end, dur, curr_time):
    if (ini.dim() > 1):
        curr_time = curr_time.unsqueeze(1)
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




class HermiteCurveQuat_torch(object):
    #formalism "wxyz"
    def __init__(self, n_batch):
        self._n_batch = n_batch
        self._q0 = torch.zeros(n_batch, 4, dtype = torch.double)
        self._w1 = torch.zeros(n_batch, 4, dtype = torch.double)
        self._w2 = torch.zeros(n_batch, 3, dtype = torch.double)
        self._w3 = torch.zeros(n_batch, 3, dtype = torch.double)

        self._qa = torch.zeros(n_batch, 4, dtype = torch.double)
        self._qb = torch.zeros(n_batch, 4, dtype = torch.double)
        self._wa = torch.zeros(n_batch, 3, dtype = torch.double)
        self._wb = torch.zeros(n_batch, 3, dtype = torch.double)

        self._q1 = torch.zeros(n_batch, 4, dtype = torch.double)
        self._q2 = torch.zeros(n_batch, 4, dtype = torch.double)

        self._duration = torch.ones(n_batch, dtype = torch.double)

    def setParams(self, id, q_a, q_b, w_a, w_b, duration): #id must be a list of indexes
        #formalism "wxyz"
        self._qa[id] = q_a
        self._qb[id] = q_b  #q3
        self._wa[id] = w_a
        self._wb[id] = w_b
        self._duration[id] = duration

        self._q0 = self._qa
        self._w1 = self._wa/3.
        self._w3 = self._wb/3.

        
        mask = torch.where(torch.linalg.norm(self._wa, dim = 1) < 1e-6, True, False)
        idx_plus = torch.nonzero(mask).squeeze().tolist()
        idx_plus = [idx_plus] if isinstance(idx_plus, int) else idx_plus
        idx_minus = torch.nonzero(~mask).squeeze().tolist()
        idx_minus = [idx_minus] if isinstance(idx_minus, int) else idx_minus

        if len(idx_plus) > 0:
            self._q1[idx_plus] = orbit_util.quat_mul(self._qa[idx_plus], 
                                                     torch.tensor([1., 0., 0., 0.], dtype = torch.double).repeat(len(idx_plus), 1))
        if len(idx_minus) > 0:
            """
            rot_vec_1 = orbit_util.quat_from_angle_axis(torch.linalg.norm(self._w1[idx_minus]) * self._wa[idx_minus]/torch.linalg.norm(self._wa[idx_minus]))
            self._q1[idx_minus] = orbit_util.quat_mul(self._qa[idx_minus],
                                                      rot_vec_1)
            """
            self._q1[idx_minus] = orbit_util.quat_mul(self._qa[idx_minus], 
                                                      util.quat_from_rot_vec(self._w1[idx_minus]))

        mask2 = torch.where(torch.linalg.norm(self._wb, dim = 1) < 1e-6, True, False)
        idx_plus2 = torch.nonzero(mask2).squeeze().tolist()
        idx_minus2 = torch.nonzero(~mask2).squeeze().tolist()

        idx_plus2 = [idx_plus2] if isinstance(idx_plus2, int) else idx_plus2
        idx_minus2 = [idx_minus2] if isinstance(idx_minus2, int) else idx_minus2

        if len(idx_plus2) > 0:
            self._q2[idx_plus2] = orbit_util.quat_mul(self._qb[idx_plus2], 
                                                      torch.tensor([1., 0., 0., 0.], dtype = torch.double).repeat(len(idx_plus2), 1))
        if len(idx_minus2) > 0:
            self._q2[idx_minus2] = orbit_util.quat_mul(self._qa[idx_minus2],
                                                       util.quat_from_rot_vec(self._w3[idx_minus2]))

        self._omega_1aa = orbit_util.quat_mul(self._q1, orbit_util.quat_inv(self._q0))
        self._omega_2aa = orbit_util.quat_mul(self._q2, orbit_util.quat_inv(self._q1))
        self._omega_3aa = orbit_util.quat_mul(self._qb, orbit_util.quat_inv(self._q2))

        self._omega_1 = orbit_util.axis_angle_from_quat(self._omega_1aa)
        self._omega_2 = orbit_util.axis_angle_from_quat(self._omega_2aa)
        self._omega_3 = orbit_util.axis_angle_from_quat(self._omega_3aa)


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
        s_in = s_in/self._duration
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)

        mask = torch.where(torch.linalg.norm(self._omega_1, dim = 1) > 1e-5, True, False)
        idx_plus = torch.nonzero(mask).squeeze().tolist()
        idx_plus = [idx_plus] if isinstance(idx_plus, int) else idx_plus

        qtmp1 = torch.tensor([1., 0., 0., 0.], dtype = torch.double).repeat(self._n_batch, 1)
        if (len(idx_plus) > 0):
            qtmp1[idx_plus] = util.quat_from_rot_vec(self._b1[idx_plus]*self._omega_1[idx_plus])

        mask2 = torch.where(torch.linalg.norm(self._omega_2, dim = 1) > 1e-5, True, False)
        idx_plus2 = torch.nonzero(mask2).squeeze().tolist()
        idx_plus2 = [idx_plus2] if isinstance(idx_plus2, int) else idx_plus2
        
        qtmp2 = torch.tensor([1., 0., 0., 0.], dtype = torch.double).repeat(self._n_batch, 1)
        if (len(idx_plus2) > 0):
            qtmp2[idx_plus2] = util.quat_from_rot_vec(self._b2[idx_plus2]*self._omega_2[idx_plus2])

        mask3 = torch.where(torch.linalg.norm(self._omega_3, dim = 1) > 1e-5, True, False)
        idx_plus3 = torch.nonzero(mask3).squeeze().tolist()
        idx_plus3 = [idx_plus3] if isinstance(idx_plus3, int) else idx_plus3

        qtmp3 = torch.tensor([1., 0., 0., 0.], dtype = torch.double).repeat(self._n_batch, 1)
        if (len(idx_plus3) > 0):
            qtmp2[idx_plus3] = util.quat_from_rot_vec(self._b2[idx_plus3]*self._omega_2[idx_plus3])



        return orbit_util.quat_mul(qtmp3, 
                                    orbit_util.quat_mul(qtmp2, 
                                                        orbit_util.quat_mul(qtmp1, self._q0)))

    def evaluate_ang_vel(self, s_in):
        s_in = s_in/self._duration
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)

        return self._omega_1 * self._bdot1 + self._omega_2 * self._bdot2 + self._omega_3 * self._bdot3

    def evaluate_ang_acc(self, s_in):
        s_in = s_in/self._duration
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)

        return self._omega_1 * self._bddot1 + self._omega_2 * self._bddot2 + self._omega_3 * self._bddot3


"""TODO: if have time see why problem when reaching a certain angle of turn
        always same angle 
class HermiteCurveQuat_torch_test(object):
    #https://dl.acm.org/doi/pdf/10.1145/218380.218486
    #quaternion formalism will be (w, x,, y, z)

    def __init__(self, n_batch):
        self._n_batch = n_batch
        self._q0 = torch.zeros(n_batch, 4, dtype = torch.double)
        self._w1 = torch.zeros(n_batch, 4, dtype = torch.double)
        self._w2 = torch.zeros(n_batch, 3, dtype = torch.double)
        self._w3 = torch.zeros(n_batch, 3, dtype = torch.double)

        self._qa = torch.zeros(n_batch, 4, dtype = torch.double)
        self._qb = torch.zeros(n_batch, 4, dtype = torch.double)
        self._wa = torch.zeros(n_batch, 3, dtype = torch.double)
        self._wb = torch.zeros(n_batch, 3, dtype = torch.double)

        self._duration = torch.ones(n_batch, dtype = torch.double)
    
    def setParams(self, id, q_a, q_b, w_a, w_b, duration): #id must be a list of indexes
        self._qa[id] = q_a
        self._qb[id] = q_b
        self._wa[id] = w_a
        self._wb[id] = w_b
        self._duration[id] = duration

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
        s_in = s_in/self._duration
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)
        res = orbit_util.quat_mul(orbit_util.quat_mul(orbit_util.quat_mul(
                                        self._q0, 
                                        util.exp_quat_map(self._w1 * self._b1)),
                                        util.exp_quat_map(self._w2 * self._b2)),
                                        util.exp_quat_map(self._w3 * self._b3))
        return res
    #TODO: check if makes sense, ang vel and ang acc
    def evaluate_ang_vel(self, s_in):
        s_in = s_in/self._duration
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)

        return self._w1 * self._bdot1 + self._w2 * self._bdot2 + self._w3* self._bdot3
    
    def evaluate_ang_acc(self, s_in):
        s_in = s_in/self._duration
        s = torch.clamp(s_in, 0., 1.)
        self._compute_basis(s)

        return self._w1 * self._bddot1 + self._w2 * self._bddot2 + self._w3* self._bddot3
"""


    
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
    def __init__(self, n_batch):
        self._n_batch = n_batch
        self._start_pos = torch.zeros(self._n_batch, 3, dtype = torch.double)
        self._end_pos = torch.zeros(self._n_batch, 3, dtype = torch.double)
        self._mid_z_pos = torch.zeros(self._n_batch, dtype = torch.double)
        self._duration = torch.zeros(self._n_batch, dtype = torch.double)
        self._st_time = torch.zeros(self._n_batch, dtype = torch.double)
        self.z_curve = QuadraticLagrangePol(self._n_batch)
    
    def setParams(self, ids, start_pos, end_pos, mid_z_pos, duration):
        assert len(ids) > 0
        self._start_pos[ids] = start_pos.clone().detach()
        self._end_pos[ids] = end_pos.clone().detach()
        self._mid_z_pos[ids] = mid_z_pos.clone().detach()
        self._duration[ids] = duration.clone().detach()

        self.z_curve.setParams(ids, start_pos[:, 2], torch.zeros(len(ids), dtype = torch.double), mid_z_pos, duration/2, end_pos[:,2], duration)

    def evaluate(self, t):
        _t = torch.clamp(t, self._st_time, self._duration)  

        s = _t/self._duration
        #if s > 1 return end pos

        #x = 0.5*((1+torch.cos(math.pi*s))*self._start_pos[:, 0] + (1-torch.cos(math.pi*s))*self._end_pos[:, 0])
        #y = 0.5*((1+torch.cos(math.pi*s))*self._start_pos[:, 1] + (1-torch.cos(math.pi*s))*self._end_pos[:, 1])

        x = 0.5*(self._start_pos[:, 0] + self._end_pos[:, 0] + torch.cos(math.pi*s)*(self._start_pos[:, 0] - self._end_pos[:, 0]))
        y = 0.5*(self._start_pos[:, 1] + self._end_pos[:, 1] + torch.cos(math.pi*s)*(self._start_pos[:, 1] - self._end_pos[:, 1]))
        z = self.z_curve.evaluate(_t)

        return torch.stack((x,y,z), dim = 1)
    
    def evaluate_first_derivative(self, t):
        _t = torch.clamp(t, self._st_time, self._duration)  
        s = _t/self._duration
        #if s > 1 return end pos

        x = 0.5*math.pi*torch.sin(math.pi*s)/self._duration*(self._end_pos[:, 0] - self._start_pos[:, 0])
        y = 0.5*math.pi*torch.sin(math.pi*s)/self._duration*(self._end_pos[:, 1] - self._start_pos[:, 1])
        
        z = self.z_curve.evaluate_first_derivative(_t)

        return torch.stack((x, y, z), dim = 1)
    
    def evaluate_second_derivative(self, t):
        _t = torch.clamp(t, self._st_time, self._duration)  
        s = _t/self._duration
        #if s > 1 return end pos

        x = 0.5*math.pi*math.pi*torch.cos(math.pi*s)/self._duration/self._duration * (self._end_pos[:, 0] - self._start_pos[:, 0])
        y = 0.5*math.pi*math.pi*torch.cos(math.pi*s)/self._duration/self._duration * (self._end_pos[:, 1] - self._start_pos[:, 1])
        z = self.z_curve.evaluate_second_derivative(_t)

        #filtering
        x = torch.where(x <  25., x,  25.)
        x = torch.where(x > -25., x, -25.)
        return torch.stack((x, y, z), dim = 1)




class QuadraticLagrangePol(object): #sizes are torhc.tensor([n_batch])
    def __init__(self, n_batch):
        self._n_batch = n_batch
        self._z0 = torch.zeros(self._n_batch, dtype = torch.double)
        self._z1 = torch.zeros(self._n_batch, dtype = torch.double)
        self._z2 = torch.zeros(self._n_batch, dtype = torch.double)
        self._t0 = torch.zeros(self._n_batch, dtype = torch.double)
        self._t1 = torch.zeros(self._n_batch, dtype = torch.double)
        self._t2 = torch.zeros(self._n_batch, dtype = torch.double)



    def setParams(self, ids, z0, t0, z1, t1, z2, t2):
        self._z0[ids] = z0.clone()
        self._z1[ids] = z1.clone()
        self._z2[ids] = z2.clone()
        self._t0[ids] = t0.clone()
        self._t1[ids] = t1.clone()
        self._t2[ids] = t2.clone() 

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

