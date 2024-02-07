import torch
torch.manual_seed(0)

import random 
random.seed(0)

import numpy as np

from pnc_pytorch.wbc.ihwbc.ihwbc import IHWBC
from pnc.wbc.ihwbc.ihwbc import IHWBC as npIHWBC


class DummyTask():
    def __init__(self, a, nb, nq_dot, dim, j = None):
        self.w_hierarchy = a*torch.ones(nb)
        if j is None:
            print("hehe")
            self.jacobian = a*torch.eye(dim, nq_dot).unsqueeze(0).repeat(nb, 1, 1)
        else:
            self.jacobian = a*j
        self.jacobian_dot_q_dot = torch.ones(nb, dim)
        self.op_cmd = torch.ones(nb, dim)

class npDummyTask():
    def __init__(self, a, nq_dot, dim, j = None):
        self.w_hierarchy = a
        if j is None:

            self.jacobian = a*np.eye(dim, nq_dot)
        else:
            self.jacobian = a*j
        self.jacobian_dot_q_dot = np.ones(dim)
        self.op_cmd = np.ones(dim)

class DummyInternalConstraint():
    def __init__(self, nb, dim, nq_dot):
        self.jacobian = torch.zeros(nb, dim, nq_dot)
        self.jacobian_dot_q_dot = torch.ones(nb, dim)
        self.jacobian[:, 0, 8] = 1.
        self.jacobian[:, 0, 8] = -1.
        self.jacobian[:, 1, 9] = 1.
        self.jacobian[:, 1, 9] = -1.

class npDummyInternalConstraint():
    def __init__(self, dim, nq_dot):
        self.jacobian = np.zeros((dim, nq_dot))
        self.jacobian_dot_q_dot = np.ones(dim)
        self.jacobian[0, 8] = 1.
        self.jacobian[0, 8] = -1.
        self.jacobian[1, 9] = 1.
        self.jacobian[1, 9] = -1.

class DummyContact():
    def __init__(self, dim, nq, nb, rf_z_max):
        self.jacobian = torch.eye( dim, nq).unsqueeze(0).repeat(nb, 1, 1)
        self.jacobian_dot_q_dot = torch.zeros(nb, dim)
        self.rf_z_max = rf_z_max * torch.ones(nb)
        self.mu = 0.3

        self.cone_constraint_mat = torch.zeros(nb, 6, dim)
        self.cone_constraint_mat[:, 0, 2] = 1.

        self.cone_constraint_mat[:, 1, 0] = 1.
        self.cone_constraint_mat[:, 1, 2] = self.mu
        self.cone_constraint_mat[:, 2, 0] = -1.
        self.cone_constraint_mat[:, 2, 2] = self.mu

        self.cone_constraint_mat[:, 3, 1] = 1.
        self.cone_constraint_mat[:, 3, 2] = self.mu
        self.cone_constraint_mat[:, 4, 1] = -1.
        self.cone_constraint_mat[:, 4, 2] = self.mu

        self.cone_constraint_mat[:, 5, 2] = -1.
        #self.cone_contraint_mat = self.cone_constraint_mat.unsqueeze(0).repeat(nb, 1,1)

        self.cone_constraint_vec = torch.zeros(nb, 6)
        self.cone_constraint_vec[:, 5] = -self.rf_z_max



class npDummyContact():
    def __init__(self, dim, nq, rf_z_max):
        self.jacobian = np.eye(dim, nq)
        self.jacobian_dot_q_dot = np.zeros(dim)
        self.rf_z_max = rf_z_max
        self.mu = 0.3

        self.cone_constraint_mat = np.zeros((6, dim))
        self.cone_constraint_mat[ 0, 2] = 1.

        self.cone_constraint_mat[1, 0] = 1.
        self.cone_constraint_mat[1, 2] = self.mu
        self.cone_constraint_mat[2, 0] = -1.
        self.cone_constraint_mat[2, 2] = self.mu

        self.cone_constraint_mat[3, 1] = 1.
        self.cone_constraint_mat[3, 2] = self.mu
        self.cone_constraint_mat[4, 1] = -1.
        self.cone_constraint_mat[4, 2] = self.mu

        self.cone_constraint_mat[5, 2] = -1.
        #self.cone_contraint_mat = self.cone_constraint_mat.unsqueeze(0).repeat(nb, 1,1)

        self.cone_constraint_vec = np.zeros(6)
        self.cone_constraint_vec[5] = -self.rf_z_max


    
# def __init__(self, sf, sa, sv, n_batch, data_save=False):  #check for sf, sa, sv types must be torch tensors: shape must be shape of x = [n_batch, shape[x]]
n_active = 10
n_q_dot = 20
n_passive = n_q_dot - n_active - 6
n_batch = 3

sa = torch.eye(n_active, n_q_dot)
sv = torch.eye(n_passive, n_q_dot)
sf = torch.eye(6, n_q_dot)

npsa = sa.numpy()
npsv = sv.numpy()
npsf = sf.numpy()

ihwbc = IHWBC(sf, sa, sv, n_batch)
npihwbc = npIHWBC(npsf, npsa, npsv)

ihwbc._lambda_q_ddot = 1.
ihwbc._lambda_rf = 1.
ihwbc._w_rf = 1.

npihwbc._lambda_q_ddot = 1.
npihwbc._lambda_rf = 1.
npihwbc._w_rf = 1.


mass_matrix = torch.eye(n_q_dot).unsqueeze(0).repeat(n_batch, 1, 1)
mass_matrix_inv = torch.eye(n_q_dot).unsqueeze(0).repeat(n_batch, 1, 1)
coriolis = torch.randn(n_q_dot).unsqueeze(0).repeat(n_batch, 1)
gravity = torch.randn(n_q_dot).unsqueeze(0).repeat(n_batch, 1)
ihwbc.update_setting(mass_matrix, mass_matrix_inv, coriolis, gravity)

npmass_matrix = mass_matrix[0, :, :].numpy()
npmass_matrix_inv = mass_matrix_inv[0, :,:].numpy()
npcoriolis = coriolis[0, :].numpy()
npgravity = gravity[0, :].numpy()
npihwbc.update_setting(npmass_matrix, npmass_matrix_inv, npcoriolis, npgravity)


jTask = torch.randn(3, n_q_dot)
npjTask = jTask.numpy()
jTask = jTask.unsqueeze(0).repeat(n_batch, 1, 1)

nptask1 = npDummyTask(1, n_q_dot, 3, npjTask)
nptask2 = npDummyTask(100, n_q_dot, 3, npjTask)
nptask_list = [nptask1, nptask2]
npihwbc._w_hierarchy = np.array([1, 2])

npic1 = npDummyInternalConstraint(2, n_q_dot)
npic_list = [npic1]

npcontact1 = npDummyContact(3, n_q_dot, 2)
npcontact2 = npDummyContact(3, n_q_dot, 0)
npcontact_list = [npcontact1, npcontact2]


task1 = DummyTask(1, n_batch, n_q_dot, 3, jTask)
task2 = DummyTask(10, n_batch, n_q_dot, 3, jTask)
task_list = [task1, task2]
ihwbc._w_hierarchy = torch.tensor([1, 2]).unsqueeze(0).repeat(n_batch, 1)

ic1 = DummyInternalConstraint(n_batch, 2, n_q_dot)
ic_list = [ic1]

contact1 = DummyContact(3, n_q_dot, n_batch, 2)
contact2 = DummyContact(3, n_q_dot, n_batch, 0)
contact_list = [contact1, contact2]





npihwbc.solve(nptask_list, npcontact_list, npic_list)
ihwbc.solve(task_list, contact_list, ic_list)

cost_mat = npihwbc.cost_mat - ihwbc.cost_mat.numpy()
cost_vec = npihwbc.cost_vec - ihwbc.cost_vec.numpy()
ineq_mat = npihwbc.ineq_mat - ihwbc.ineq_mat.numpy()
ineq_vec = npihwbc.ineq_vec - ihwbc.ineq_vec.numpy()
eq_mat = npihwbc.eq_mat - ihwbc.eq_mat.numpy()
eq_vec = npihwbc.eq_vec - ihwbc.eq_vec.numpy()
#print("cost_mat", cost_mat)
#print("cost_vec", cost_vec)
#print("ineq_mat", ineq_mat)
#print("ineq_vec", ineq_vec)
#print("eq_mat", eq_mat)
#print("eq_vec", eq_vec)





check_cost_mat = not np.any(cost_mat)
print("check_cost_mat", check_cost_mat)

check_cost_vec = not np.any(cost_vec)
print("check_cost_vec", check_cost_vec)

check_ineq_mat = not np.any(ineq_mat)
print("check_ineq_mat", check_ineq_mat)

check_ineq_vec = not np.any(ineq_vec)
print("check_ineq_vec", check_ineq_vec)

check_eq_mat = not np.any(eq_mat)
print("check_eq_mat", check_eq_mat)

check_eq_vec = not np.any(eq_vec)
print("check_eq_vec", check_eq_vec)


torch.set_printoptions(threshold=10_000)
#print(torch.linalg.matrix_rank(ihwbc.cost_mat), ihwbc.cost_mat.shape)

if (check_cost_mat == False): 
    print("cost_mat ", cost_mat, np.linalg.norm(cost_mat))

if (check_cost_vec == False): 
    print("cost_mat ", cost_vec, np.linalg.norm(cost_vec))

if (check_ineq_mat == False): 
    print("cost_mat ", ineq_mat, np.linalg.norm(ineq_mat))

if (check_ineq_vec == False): 
    print("cost_mat ", ineq_vec, np.linalg.norm(ineq_vec))

if (check_eq_mat == False): 
    print("cost_mat ", eq_mat, np.linalg.norm(ineq_mat))

if (check_eq_vec == False): 
    print("cost_mat ", eq_vec, np.linalg.norm(ineq_vec))

print(ihwbc.sol - npihwbc.sol)
print(torch.linalg.matrix_norm(ihwbc.sol - npihwbc.sol))
