from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import json
import configparser

import multiprocessing as mp
from tqdm import tqdm
from util import orbit_util

import torch 


def pretty_print(ob):
    print(json.dumps(ob, indent=4))



# Function to read configuration from file
def read_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def euler_to_rot(angles):
    # Euler ZYX to Rot
    # Note that towr has (x, y, z) order
    x = angles[0]
    y = angles[1]
    z = angles[2]
    ret = np.array([
        np.cos(y) * np.cos(z),
        np.cos(z) * np.sin(x) * np.sin(y) - np.cos(x) * np.sin(z),
        np.sin(x) * np.sin(z) + np.cos(x) * np.cos(z) * np.sin(y),
        np.cos(y) * np.sin(z),
        np.cos(x) * np.cos(z) + np.sin(x) * np.sin(y) * np.sin(z),
        np.cos(x) * np.sin(y) * np.sin(z) - np.cos(z) * np.sin(x), -np.sin(y),
        np.cos(y) * np.sin(x),
        np.cos(x) * np.cos(y)
    ]).reshape(3, 3)
    return np.copy(ret)


def quat_to_rot(quat):
    """
    Parameters
    ----------
    quat (np.array): scalar last quaternion

    Returns
    -------
    ret (np.array): SO3

    """
    return np.copy((R.from_quat(quat)).as_matrix())


def rot_to_quat(rot):
    """
    Parameters
    ----------
    rot (np.array): SO3

    Returns
    -------
    quat (np.array): scalar last quaternion

    """
    return np.copy(R.from_matrix(rot).as_quat())


def quat_to_exp(quat):
    img_vec = np.array([quat[0], quat[1], quat[2]])
    w = quat[3]
    theta = 2.0 * np.arcsin(
        np.sqrt(img_vec[0] * img_vec[0] + img_vec[1] * img_vec[1] +
                img_vec[2] * img_vec[2]))

    if np.abs(theta) < 1e-4:
        return np.zeros(3)
    ret = img_vec / np.sin(theta / 2.0)

    return np.copy(ret * theta)

def quat_to_exp_pytorch(quat):
    #formalism is (x, y, z, w)
    batch = quat.shape[0]
    img_vec = torch.stack((quat[:, 0], quat[:, 1], quat[:, 2]), dim = 1)
    w = quat[:, 3]
    theta = 2.0 * torch.asin(
        torch.sqrt(img_vec[:, 0] * img_vec[:, 0] + img_vec[:, 1] * img_vec[:, 1] +
                img_vec[:, 2] * img_vec[:, 2]))

    #ret = torch.zeros_like(img_vec)
    ret = torch.where(torch.abs(theta).unsqueeze(1) < 1e-4, torch.zeros(batch, 3, dtype = torch.double), 
                                                            img_vec / torch.sin(theta / 2.0).unsqueeze(1))

    return torch.clone(ret * theta.unsqueeze(1))


def exp_to_quat(exp):
    theta = np.sqrt(exp[0] * exp[0] + exp[1] * exp[1] + exp[2] * exp[2])
    ret = np.zeros(4)
    if theta > 1e-4:
        ret[0] = np.sin(theta / 2.0) * exp[0] / theta
        ret[1] = np.sin(theta / 2.0) * exp[1] / theta
        ret[2] = np.sin(theta / 2.0) * exp[2] / theta
        ret[3] = np.cos(theta / 2.0)
    else:
        ret[0] = 0.5 * exp[0]
        ret[1] = 0.5 * exp[1]
        ret[2] = 0.5 * exp[2]
        ret[3] = 1.0
    return np.copy(ret)

def exp_to_quat_pytorch(exp):
    theta = torch.sqrt(exp[:, 0]*exp[:, 0] + exp[:, 1]*exp[:, 1] + exp[:, 2]*exp[:, 2])

    ret = torch.where(theta.unsqueeze(1) > 1e-4, torch.stack((torch.sin(theta/2.0)*exp[:, 0] / theta,
                                                              torch.sin(theta/2.0)*exp[:, 1] / theta,
                                                              torch.sin(theta/2.0)*exp[:, 2] / theta,
                                                              torch.cos(theta/2.0)), dim = 1), 
                                                  torch.stack((0.5*exp[:, 0], 0.5*exp[:, 1], 0.5*exp[:, 2], torch.ones(exp.shape[0])), dim = 1))
    return torch.clone(ret)


def weighted_pinv(A, W, rcond=1e-15):
    return np.dot(
        W,
        np.dot(A.transpose(),
               np.linalg.pinv(np.dot(np.dot(A, W), A.transpose()), rcond)))

def weighted_pinv_pytorch(A, W, rtol=1e-15):
    return torch.bmm(
        W,
        torch.bmm(A.transpose(1,2),
               torch.linalg.pinv(torch.bmm(torch.bmm(A, W), A.transpose(1,2)), rtol=rtol)))

def get_sinusoid_trajectory(start_time, mid_point, amp, freq, eval_time):
    dim = amp.shape[0]
    p, v, a = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    p = amp * np.sin(2 * np.pi * freq * (eval_time - start_time)) + mid_point
    v = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq *
                                        (eval_time - start_time))
    a = -amp * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq *
                                              (eval_time - start_time))

    return p, v, a


def normalize_data(data):
    mean = np.mean(np.stack(data, axis=0), axis=0)
    std = np.std(np.stack(data, axis=0), axis=0)

    return mean, std, normalize(data, mean, std)


def normalize(x, mean, std):
    assert std.shape == mean.shape
    if type(x) is list:
        assert x[0].shape == mean.shape
        ret = []
        for val in x:
            ret.append((val - mean) / std)
        return ret
    else:
        assert x.shape == mean.shape
        return (x - mean) / std


def denormalize(x, mean, std):
    assert std.shape == mean.shape
    if type(x) is list:
        assert x[0].shape == mean.shape
        ret = []
        for val in x:
            ret.append(val * std + mean)
        return ret
    else:
        assert x.shape == mean.shape
        return x * std + mean


def print_attrs(ob):
    attr = vars(ob)
    print(", \n".join("%s: %s" % item for item in attr.items()))


def try_multiprocess(args_list, num_cpu, f, max_timeouts=1):
    """
    Multiprocessing wrapper function.
    """
    if max_timeouts == 0:
        return None

    if num_cpu == 1:
        return [f(args_list)]
    else:
        pool = mp.Pool(processes=num_cpu,
                       maxtasksperchild=1,
                       initargs=(mp.RLock(), ),
                       initializer=tqdm.set_lock)
        pruns = []
        for i in range(num_cpu):
            rseed = np.random.randint(1000000)
            pruns.append(pool.apply_async(f, args=(args_list + [rseed, i], )))
        try:
            results = [p.get(timeout=36000) for p in pruns]
        except Exception as e:
            print(str(e))
            print('WARNING: error raised in multiprocess, trying again')

            pool.close()
            pool.terminate()
            pool.join()

            return try_multiprocess(args_list, num_cpu, f, max_timeouts - 1)

        pool.close()
        pool.terminate()
        pool.join()

    return results


def prevent_quat_jump(quat_des, quat_act):
    #print("quat_des:",quat_des)
    # print("quat_act:",quat_act)
    a = quat_des - quat_act
    b = quat_des + quat_act
    if np.linalg.norm(a) > np.linalg.norm(b):
        new_quat_act = -quat_act
    else:
        new_quat_act = quat_act

    return new_quat_act

def prevent_quat_jump_pytorch(quat_des, quat_act):
    # print("quat_des:",quat_des)
    # print("quat_act:",quat_act)
    a = quat_des - quat_act
    b = quat_des + quat_act
    """CARLOS"""
    #probaly better to change to torch.linalg.vector_norm()
    if torch.linalg.norm(a) > torch.linalg.norm(b):
        new_quat_act = -quat_act
    else:
        new_quat_act = quat_act

    return new_quat_act

def is_colliding_3d(start, goal, min, max, threshold, N):
    for i in range(3):
        for j in range(N):
            p = start[i] + (goal[i] - start[i]) * j / N
            if min[i] + np.abs(threshold[i]) <= p and p <= max[i] - np.abs(
                    threshold[i]):
                return True
    return False


class GridLocation(object):
    def __init__(self, delta):
        """
        Parameters
        ----------
        delta (np.array): 1d array
        """
        self._dim = delta.shape[0]
        self._delta = np.copy(delta)

    def get_grid_idx(self, pos):
        """
        Parameters
        ----------
        pos (np.array): 1d array

        Returns
        -------
        v (double or tuple): idx
        """
        v = np.zeros(self._dim, dtype=int)
        for i in range(self._dim):
            v[i] = pos[i] // self._delta[i]

        if self._dim == 1:
            return v[0]
        else:
            return tuple(v)

    def get_boundaries(self, idx):
        """
        Parameters
        ----------
        idx (np.array): 1d array of integer

        Returns
        -------
        v (np.array): 1d array of boundaries [min, max, min, max]
        """

        bds = np.zeros(self._dim * 2, dtype=float)
        for i in range(self._dim):
            bds[2 * i] = idx[i] * self._delta[i]
            bds[2 * i + 1] = (idx[i] + 1) * self._delta[i]

        return bds

    def get_center(self, idx):
        """
        Parameters
        ----------
        idx (np.array): 1d array of integer

        Returns
        -------
        c (np.array): center
        """

        if self._dim == 1:
            bds = self.get_boundaries(idx)
            return (bds[0] + bds[1]) / 2.
        else:
            bds = self.get_boundaries(idx)
            return np.array([(bds[0] + bds[1]) / 2., (bds[2] + bds[3]) / 2.])

def MakeHorizontalDirX(rot):
    """
    Returns a rotation matrix parallel to ground ==> z' = [0 ,0 ,1]
    With with the same x direction.
    That is we set z, we project x into z plane and obtain x'. Compute y' to be ortonormal
    """
    new_rot = rot.clone().detach()
    new_rot[:, 0, 2] = 0.
    new_rot[:, 1, 2] = 0.
    new_rot[:, 2, 2] = 1.
    new_rot[:, 2, 0] = 0.
    new_rot[:, 2, 1] = 0.
    new_rot[:, :, 0] = new_rot[:, : , 0] / torch.linalg.norm(new_rot[:, :, 0], dim = 1)[:, None]
    new_rot[:, :, 1] = torch.linalg.cross(new_rot[:, :, 2], new_rot[:, :, 0])

    return new_rot

def rotationZ(theta, inMatrix):
    """
    theta input is in degrees
    """
    assert len(inMatrix.shape) == 3
    theta_radians = torch.deg2rad(theta)
    cos_theta = torch.cos(theta_radians)
    sin_theta = torch.sin(theta_radians)

    # Construct rotation matrix
    rotation_matrix = torch.stack([cos_theta, -sin_theta, torch.zeros_like(theta),
                                   sin_theta, cos_theta, torch.zeros_like(theta),
                                   torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1).reshape(-1, 3, 3) 
    rotation_matrix = rotation_matrix.to(inMatrix.dtype) 
    res = torch.bmm(rotation_matrix, inMatrix)
    return res


def log_quat_map(quat):
    #assumes quat is normalized
    #logarithm of batched quat as in:
    # https://doi.org/10.1002/(SICI)1099-1778(199601)7:1<43::AID-VIS136>3.0.CO;2-T
    #could use eps and torch.where instead of nan_to_num

    batch = quat.shape[0]
    theta = torch.acos(quat[:, 0])
    sin = torch.linalg.vector_norm(quat[:, 1:4], dim = 1)
    #theta = torch.where(sin > eps, theta, 0.)
    #sin = torch.where(sin > eps, sin, 1.)
    res = theta.unsqueeze(1)/sin.unsqueeze(1) * quat[:, 1:4]
    #res = torch.cat((torch.zeros(batch).unsqueeze(1), res), dim = 1)
    return torch.nan_to_num(res)

def exp_quat_map(v):
    # https://doi.org/10.1002/(SICI)1099-1778(199601)7:1<43::AID-VIS136>3.0.CO;2-T
    # must be batch x 3d vector
    #could use eps and torch.where instead of nan_to_num

    norm = torch.linalg.vector_norm(v, dim = 1)
    norm = torch.where(norm > 1e-5, norm, 0.)
    res = torch.cat((torch.cos(norm).unsqueeze(1), (torch.sin(norm)/norm).unsqueeze(1) * v), dim = 1)
    return torch.nan_to_num(res)


def quat_mul_xyzw(q1, q2):
    return orbit_util.convert_quat(orbit_util.quat_mul(
                          orbit_util.convert_quat(q1, to = "wxyz"),
                          orbit_util.convert_quat(q2, to = "wxyz")))

def quat_inv_xyzw(q):
    return orbit_util.convert_quat(orbit_util.quat_inv(orbit_util.convert_quat(q, to = "wxyz")))