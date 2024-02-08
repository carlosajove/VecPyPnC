import sys
import numpy as np
import torch
import time
np.set_printoptions(precision=2, threshold=sys.maxsize)
torch.set_printoptions(precision = 2, threshold=10000)

from scipy.linalg import block_diag
from qpsolvers import solve_qp

from pnc_pytorch.wbc.ihwbc.qpth.qp import QPFunction

from util import util
from pnc.data_saver import DataSaver

def printvar(a, b):
    print("np ", a, "\n", b, b.shape, "\n", b.dtype)

def printeps(name, array):
    eps = 1e-8
    masked_array = np.where(array < eps, 0, array)
    num_non_zero_after = np.count_nonzero(masked_array)
    if(num_non_zero_after > 0):
        print("="*80, f"\n {name}:, \n num non zero: {num_non_zero_after} \n", 
                "=" * 80 )


# ERROR FOUND IN COST. 
# ERROR COMES WHEN ADDING VALUES
# ERROR IS OF AROUND 1E-4
# NO OTHER ERROR FOUND
# HOWEVER NOT TESTED INEQ EQ WITH TRQ LIMITS

class IHWBC(object):
    """
    Implicit Hierarchy Whole Body Control
    ------------------
    Usage:
        update_setting --> solve
    """
    def __init__(self, sf, sa, sv, data_save=False):

        self._n_q_dot = sa.shape[1]
        self._n_active = sa.shape[0]
        self._n_passive = sv.shape[0]

        self._sf = sf
        self._snf = np.concatenate((np.zeros(
            (self._n_active + self._n_passive, 6)),
                                    np.eye(self._n_active + self._n_passive)),
                                   axis=1)
        self._sa = sa
        self._sv = sv

        self._trq_limit = None
        self._lambda_q_ddot = 0.
        self._lambda_rf = 0.
        self._w_rf = 0.
        self._w_hierarchy = 0.

        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()

    @property
    def trq_limit(self):
        return self._trq_limit

    @property
    def lambda_q_ddot(self):
        return self._lambda_q_ddot

    @property
    def lambda_rf(self):
        return self._lambda_rf

    @property
    def w_hierarchy(self):
        return self._w_hierarchy

    @property
    def w_rf(self):
        return self._w_rf

    @trq_limit.setter
    def trq_limit(self, val):
        assert val.shape[0] == self._n_active
        self._trq_limit = np.copy(val)

    @lambda_q_ddot.setter
    def lambda_q_ddot(self, val):
        self._lambda_q_ddot = val

    @lambda_rf.setter
    def lambda_rf(self, val):
        self._lambda_rf = val

    @w_hierarchy.setter
    def w_hierarchy(self, val):
        self._w_hierarchy = val

    @w_hierarchy.setter
    def w_rf(self, val):
        self._w_rf = val

    def update_setting(self, mass_matrix, mass_matrix_inv, coriolis, gravity):
        self._mass_matrix = np.copy(mass_matrix)
        self._mass_matrix_inv = np.copy(mass_matrix_inv)
        self._coriolis = np.copy(coriolis)
        self._gravity = np.copy(gravity)

    def solve(self,
              task_list,
              contact_list,
              internal_constraint_list,
              rf_des=None,
              verbose=False):
        """
        Parameters
        ----------
        task_list (list of Task):
            Task list
        contact_list (list of Contact):
            Contact list
        internal_constraint_list (list of InternalConstraint):
            Internal constraint list
        rf_des (np.ndarray):
            Reaction force desired
        verbose (bool):
            Printing option

        Returns
        -------
        joint_trq_cmd (np.array):
            Joint trq cmd
        joint_acc_cmd (np.array):
            Joint acc cmd
        sol_rf (np.array):
            Reaction force
        """

        # ======================================================================
        # Internal Constraint
        #   Set ni, jit_lmd_jidot_qdot, sa_ni_trc_bar_tr, and b_internal_constraint
        # ======================================================================
        if len(internal_constraint_list) > 0:
            ji = np.concatenate(
                [ic.jacobian for ic in internal_constraint_list], axis=0)
            jidot_qdot = np.concatenate(
                [ic.jacobian_dot_q_dot for ic in internal_constraint_list],
                axis=0)
            lmd = np.linalg.pinv(
                np.dot(np.dot(ji, self._mass_matrix_inv), ji.transpose()))
            ji_bar = np.dot(np.dot(self._mass_matrix_inv, ji.transpose()), lmd)
            ni = np.eye(self._n_q_dot) - np.dot(ji_bar, ji)
            jit_lmd_jidot_qdot = np.squeeze(
                np.dot(np.dot(ji.transpose(), lmd), jidot_qdot))
            sa_ni_trc = np.dot(self._sa, ni)[:, 6:]
            sa_ni_trc_bar = util.weighted_pinv(sa_ni_trc,
                                               self._mass_matrix_inv[6:, 6:])
            sa_ni_trc_bar_tr = sa_ni_trc_bar.transpose()
            b_internal_constraint = True
        else:
            ni = np.eye(self._n_q_dot)
            jit_lmd_jidot_qdot = np.zeros(self._n_q_dot)
            sa_ni_trc_bar = np.eye(self._n_active)
            sa_ni_trc_bar_tr = sa_ni_trc_bar.transpose()
            b_internal_constraint = False


        self.n_batch = 3
        self.t_mass_matrix = torch.from_numpy(self._mass_matrix).expand(self.n_batch, -1, -1)
        self.t_mass_matrix_inv = torch.from_numpy(self._mass_matrix_inv).expand(self.n_batch, -1, -1)
        self.t_sa = torch.from_numpy(self._sa).expand(self.n_batch, -1, -1)

        if len(internal_constraint_list) > 0:  
            t_ji = torch.cat(
                [torch.from_numpy(ic.jacobian).expand(self.n_batch, -1, -1)
                 for ic in internal_constraint_list], axis=1)     #ic.jacobian: [n_batch, etc]
            t_jidot_qdot = torch.cat(
                [torch.from_numpy(ic.jacobian_dot_q_dot).expand(self.n_batch, -1)
                for ic in internal_constraint_list],
                axis=1)

            t_lmd = torch.linalg.pinv(
                torch.bmm(torch.bmm(t_ji, self.t_mass_matrix_inv), t_ji.transpose(1,2)))

            t_ji_bar = torch.bmm(torch.bmm(self.t_mass_matrix_inv, t_ji.transpose(1,2)), t_lmd)
            t_ni = torch.eye(self._n_q_dot).unsqueeze(0).repeat(self.n_batch, 1, 1) - torch.bmm(t_ji_bar, t_ji)

            t_jit_lmd_jidot_qdot = torch.squeeze(
                torch.matmul(torch.bmm(t_ji.transpose(1,2), t_lmd), t_jidot_qdot.unsqueeze(2)))
            t_sa_ni_trc = torch.bmm(self.t_sa, t_ni)[:, :, 6:]

            #TODO: check util funciton
            t_sa_ni_trc_bar = util.weighted_pinv_pytorch(t_sa_ni_trc,
                                            self.t_mass_matrix_inv[:, 6:, 6:])
            t_sa_ni_trc_bar_tr = t_sa_ni_trc_bar.transpose(1, 2)
            t_b_internal_constraint = True
        else:
            t_ni = torch.eye(self._n_q_dot).unsqueeze(0).repeat(self.n_batch, 1, 1)
            t_jit_lmd_jidot_qdot = torch.zeros(self._n_q_dot).unsqueeze(0).repeat(self.n_batch, 1)
            t_sa_ni_trc_bar = torch.eye(self._n_active).unsqueeze(0).repeat(self._n_active, 1, 1)
            t_sa_ni_trc_bar_tr = t_sa_ni_trc_bar.transpose(1,2)
            b_internal_constraint = False

        """
        printeps("ji", ji - t_ji[0].numpy())
        printeps("jidot_q_dot", jidot_qdot - t_jidot_qdot[0].numpy())
        printeps("lmd", lmd - t_lmd[0].numpy())
        printeps("ji_bar", ji_bar - t_ji_bar[0].numpy())
        printeps("ni", ni - t_ni[0].numpy())
        printeps("jit_lmd_jidot_qdot", jit_lmd_jidot_qdot - t_jit_lmd_jidot_qdot[0].numpy())
        printeps("sa_ni_trc", sa_ni_trc - t_sa_ni_trc[0].numpy())
        printeps("sa_ni_trc_bar", sa_ni_trc_bar - t_sa_ni_trc_bar[0].numpy())
        printeps("sa_ni_trc_bar_tr", sa_ni_trc_bar_tr - t_sa_ni_trc_bar_tr[0].numpy())
        """
        #time.sleep(10)
        # exit()
        
        # ======================================================================
        # Cost
        # ======================================================================
        #print("\n", "%"*20, "  NEW  ", "%"*20, "\n")
        cost_t_mat = np.zeros((self._n_q_dot, self._n_q_dot))
        cost_t_vec = np.zeros(self._n_q_dot)


        t_cost_t_mat = torch.zeros(self.n_batch, self._n_q_dot, self._n_q_dot)
        t_cost_t_vec = torch.zeros(self.n_batch, self._n_q_dot)
        self.t_w_hierarchy = torch.from_numpy(self._w_hierarchy).expand(self.n_batch, -1)
        #printeps("w :", self._w_hierarchy - self.t_w_hierarchy[0].numpy())
        for i, task in enumerate(task_list):
            j = task.jacobian
            j_dot_q_dot = task.jacobian_dot_q_dot
            x_ddot = task.op_cmd
            if verbose:
                print("====================")
                print(task.target_id, " task")
                task.debug()

            t_j = torch.from_numpy(task.jacobian).expand(self.n_batch, -1, -1)
            t_j_dot_q_dot = torch.from_numpy(task.jacobian_dot_q_dot).expand(self.n_batch, -1)
            t_x_ddot = torch.from_numpy(task.op_cmd).expand(self.n_batch, -1)

            """
            printeps("j", j - t_j[0].numpy())
            printeps("t_j_dot_q_dot", j_dot_q_dot-t_j_dot_q_dot[0].numpy())
            printeps("x_ddot", x_ddot - t_x_ddot[0].numpy())
            """
            a = self._w_hierarchy[i] * np.dot(j.transpose(), j)
            b = self._w_hierarchy[i] * np.dot(
                (j_dot_q_dot - x_ddot).transpose(), j)
            t_a = self.t_w_hierarchy[:,i].unsqueeze(1).unsqueeze(1) * torch.bmm(t_j.transpose(1,2), t_j)
            t_b = self.t_w_hierarchy[:,i].unsqueeze(1) * torch.matmul(
                (t_j_dot_q_dot - t_x_ddot).unsqueeze(1), t_j).squeeze()
            

            #printvar("a", a)
            #printvar("a_t", t_a[0])
            #printeps("a eps", a-t_a[0].numpy())

            cost_t_mat += self._w_hierarchy[i] * np.dot(j.transpose(), j)
            cost_t_vec += self._w_hierarchy[i] * np.dot(
                (j_dot_q_dot - x_ddot).transpose(), j)

            t_cost_t_mat += t_a
            t_cost_t_vec += self.t_w_hierarchy[:,i].unsqueeze(1) * torch.matmul(
                (t_j_dot_q_dot - t_x_ddot).unsqueeze(1), t_j).squeeze()
            """
            printeps("cost_t_mat", cost_t_mat-t_cost_t_mat[0].numpy())
            printeps("cost_t_vec", cost_t_vec - t_cost_t_vec[0].numpy())
            """
        # cost_t_mat += self._lambda_q_ddot * np.eye(self._n_q_dot)
        #printvar("cost_t_mat", cost_t_mat)

        cost_t_mat += self._lambda_q_ddot * self._mass_matrix
        #printvar("cost_t_mat 2 ", cost_t_mat)
        #printvar("cost t vec ", cost_t_vec  )
     
        if contact_list is not None:
            uf_mat = np.array(
                block_diag(
                    *[contact.cone_constraint_mat
                      for contact in contact_list]))
            uf_vec = np.concatenate(
                [contact.cone_constraint_vec for contact in contact_list])
            contact_jacobian = np.concatenate(
                [contact.jacobian for contact in contact_list], axis=0)

            assert uf_mat.shape[0] == uf_vec.shape[0]
            assert uf_mat.shape[1] == contact_jacobian.shape[0]
            dim_cone_constraint, dim_contacts = uf_mat.shape

            cost_rf_mat = (self._lambda_rf + self._w_rf) * np.eye(dim_contacts)
            if rf_des is None:
                rf_des = np.zeros(dim_contacts)
            cost_rf_vec = -self._w_rf * np.copy(rf_des)

            cost_mat = np.array(block_diag(
                cost_t_mat, cost_rf_mat))  # (nqdot+nc, nqdot+nc)
            cost_vec = np.concatenate([cost_t_vec, cost_rf_vec])  # (nqdot+nc,)

        else:
            dim_contacts = dim_cone_constraint = 0
            cost_mat = np.copy(cost_t_mat)
            cost_vec = np.copy(cost_t_vec)

        #printvar("cost_t_mat 3", cost_t_mat)

        ###################################
        # TORCH COST
        ###################################
        t_cost_t_mat = torch.zeros(self.n_batch, self._n_q_dot, self._n_q_dot).to(torch.float64)
        t_cost_t_vec = torch.zeros(self.n_batch, self._n_q_dot).to(torch.float64)

        #the following must be batched torch tensors:
        #task.jacobian
        #task.jacobian_dot_q_dot
        #task.op_cmd
        #self._w_hierarchy
        #task.op_cmd
        for i, task in enumerate(task_list):
            t_j = torch.from_numpy(task.jacobian).expand(self.n_batch, -1, -1)
            t_j_dot_q_dot = torch.from_numpy(task.jacobian_dot_q_dot).expand(self.n_batch, -1)
            t_x_ddot = torch.from_numpy(task.op_cmd).expand(self.n_batch, -1)
            if verbose:
                print("====================")
                print(task.target_id, " task")
                task.debug()
            """
            print("IHWBC")
            printvar("w", self._w_hierarchy)
            printvar("jdotqdot", j_dot_q_dot)
            printvar("xddot", x_ddot)
            printvar("jacobian", j)
            """
            t_cost_t_mat += self.t_w_hierarchy[:,i].unsqueeze(1).unsqueeze(1).to(torch.float64) * torch.bmm(t_j.transpose(1,2), t_j).to(torch.float64)
            t_cost_t_vec += self.t_w_hierarchy[:,i].unsqueeze(1).to(torch.float64) * torch.matmul(
                (t_j_dot_q_dot - t_x_ddot).unsqueeze(1), t_j).to(torch.float64).squeeze()

            #print("t_cost_t_mat", t_cost_t_mat.dtype)

        # cost_t_mat += self._lambda_q_ddot * np.eye(self._n_q_dot)
        #TODO: check why uses mass matrix
        #printvar("cost_t_mat", cost_t_mat[0])
        t_cost_t_mat += self._lambda_q_ddot * self.t_mass_matrix.to(torch.float64)

        #printvar("cost_t_mat 2 ", cost_t_mat[0])


        #contact.cone_contraint_vec: torch.tensor([n_batch, 6])
        #contact.constraint_mat: torch.tensor([n_batch, 6, dim_contact)
        #contact.jacobian: torch.tensor([n_batch, dim_contact, robot.n_q])
        if contact_list is not None:   

            """
            CARLOS: For now use this to make batched block diagram
            """

            t_uf_mat_list = []
            for i in range(self.n_batch):
                t_uf_mat_list.append(torch.block_diag(*[torch.from_numpy(contact.cone_constraint_mat)
                    for contact in contact_list]))
            t_uf_mat = torch.stack([*t_uf_mat_list], axis = 0)
            """end batched """

            t_uf_vec = torch.cat(
                [torch.from_numpy(contact.cone_constraint_vec).expand(self.n_batch, -1)
                for contact in contact_list], axis = 1)
            t_contact_jacobian = torch.cat(
                [torch.from_numpy(contact.jacobian).expand(self.n_batch, -1, -1) 
                for contact in contact_list], axis=1)

            assert t_uf_mat.shape[1] == t_uf_vec.shape[1]
            assert t_uf_mat.shape[2] == t_contact_jacobian.shape[1]

            t_dim_cone_constraint = t_uf_mat.shape[1]
            t_dim_contacts = t_uf_mat.shape[2]

            t_cost_rf_mat = (self._lambda_rf + self._w_rf) * torch.eye(t_dim_contacts)
            #doesn't need to be batched, since params don't change between sims

            #TODO: make sure that i fwe don't have desired reaction forces, just put 0 instead
            t_rf_des = torch.from_numpy(rf_des).expand(self.n_batch, -1)
            if t_rf_des is None: #TODO: Make sure 
                t_rf_des = torch.zeros(self.n_batch, t_dim_contacts)
            
            t_cost_rf_vec = -self._w_rf * torch.clone(t_rf_des).detach()#rf_des torch.tensor([n_batch, ...])

            t_cost_mat_list = []
            for i in range(self.n_batch):
                t_cost_mat_list.append(torch.block_diag(t_cost_t_mat[i], t_cost_rf_mat)) # (nqdot+nc, nqdot+nc)
            t_cost_mat = torch.stack([*t_cost_mat_list], axis = 0)
            


            t_cost_vec = torch.cat([t_cost_t_vec, t_cost_rf_vec], axis=1)  # (nqdot+nc,)

        else:
            dim_contacts = dim_cone_constraint = 0
            t_cost_mat = torch.clone(t_cost_t_mat).detach()
            t_cost_vec = torch.clone(t_cost_t_vec).detach()

        """
        printeps("cost_t_mat", cost_t_mat-t_cost_t_mat[0].numpy())
        printeps("cost_t_vec", cost_t_vec - t_cost_t_vec.numpy())
        printeps("cost_rf_mat", cost_rf_mat - t_cost_rf_mat.numpy())
        printeps("cost_rf_vec", cost_rf_vec - t_cost_rf_vec.numpy())
        printeps("cost_mat", cost_mat -t_cost_mat.numpy())
        printeps("cost_vec", cost_vec - t_cost_vec.numpy())
        """

        # if verbose:
        # print("==================================")
        # np.set_printoptions(precision=4)
        # print("cost_t_mat")
        # print(cost_t_mat)
        # print("cost_t_vec")
        # print(cost_t_vec)
        # print("cost_rf_mat")
        # print(cost_rf_mat)
        # print("cost_rf_vec")
        # print(cost_rf_vec)
        # print("cost_mat")
        # print(cost_mat)
        # print("cost_vec")
        # print(cost_vec)

        # ======================================================================
        # Equality Constraint
        # ======================================================================

        if contact_list is not None:
            eq_floating_mat = np.concatenate(
                (np.dot(self._sf, self._mass_matrix),
                 -np.dot(self._sf,
                         np.dot(contact_jacobian, ni).transpose())),
                axis=1)  # (6, nqdot+nc)
            if b_internal_constraint:
                eq_int_mat = np.concatenate(
                    (ji, np.zeros(
                        (ji.shape[0], dim_contacts))), axis=1)  # (2, nqdot+nc)
                eq_int_vec = np.zeros(ji.shape[0])
        else:
            eq_floating_mat = np.dot(self._sf, self._mass_matrix)
            if b_internal_constraint:
                eq_int_mat = np.copy(ji)
                eq_int_vec = np.zeros(ji.shape[0])
        eq_floating_vec = -np.dot(
            self._sf, np.dot(ni.transpose(), (self._coriolis + self._gravity)))

        #printvar("eq_floating_mat", eq_floating_mat)
        #printvar("eq_int_mat", eq_int_mat)
        #printvar("eq_floaating_vec", eq_floating_vec)
        #printvar("eq_int_vec", eq_int_vec)

        if b_internal_constraint:
            eq_mat = np.concatenate((eq_floating_mat, eq_int_mat), axis=0)
            eq_vec = np.concatenate((eq_floating_vec, eq_int_vec), axis=0)
        else:
            eq_mat = np.copy(eq_floating_mat)
            eq_vec = np.copy(eq_floating_vec)

        #######
        # TORCH eq constraint
        ####### 
        self.t_sf = torch.from_numpy(self._sf).expand(self.n_batch, -1, -1)
        self.t_coriolis = torch.from_numpy(self._coriolis).expand(self.n_batch, -1)
        self.t_gravity = torch.from_numpy(self._gravity).expand(self.n_batch, -1)

        if contact_list is not None:
            t_eq_floating_mat = torch.cat(
                (torch.bmm(self.t_sf, self.t_mass_matrix),
                 -torch.bmm(self.t_sf,torch.bmm(t_contact_jacobian, t_ni).transpose(1,2))),
                dim=2)  # (6, nqdot+nc)
            if b_internal_constraint:
                t_eq_int_mat = torch.cat(
                    (t_ji, torch.zeros(self.n_batch, t_ji.shape[1], t_dim_contacts)), dim=2)  # (2, nqdot+nc)
                t_eq_int_vec = torch.zeros(self.n_batch, t_ji.shape[1])
        else:
            t_eq_floating_mat = torch.bmm(self.t_sf, self.t_mass_matrix)
            if b_internal_constraint:
                t_eq_int_mat = torch.clone(t_ji).detach()
                t_eq_int_vec = torch.zeros(self.n_batch, t_ji.shape[1])
        #printvar("ni^T", ni.transpose(1,2))
        #printvar("coriolis", self._coriolis)
        #printvar("grav", self._gravity)
        #printvar("unsqueeze", (self._coriolis + self._gravity).unsqueeze(2))
        #print(torch.matmul(ni.transpose(1,2), (self._coriolis + self._gravity).unsqueeze(2)))
        t_eq_floating_vec = -torch.matmul(
            self.t_sf, torch.matmul(t_ni.transpose(1,2), (self.t_coriolis + self.t_gravity).unsqueeze(2))).squeeze()

        #printvar("eq_floating_mat", eq_floating_mat)
        #printvar("eq_int_mat", eq_int_mat)
        #printvar("eq_floaating_vec", eq_floating_vec)
        #printvar("eq_int_vec", eq_int_vec)
        if b_internal_constraint:
            t_eq_mat = torch.cat((t_eq_floating_mat, t_eq_int_mat), dim=1)
            t_eq_vec = torch.cat((t_eq_floating_vec, t_eq_int_vec), dim=1)
        else:
            t_eq_mat = torch.clone(t_eq_floating_mat).detach()
            t_eq_vec = torch.clone(t_eq_floating_vec).detach()
        """
        printeps("eq mat", eq_mat - t_eq_mat[0].numpy())
        printeps("eq_vec", eq_vec - t_eq_vec[0].numpy() )
        """
        # ======================================================================
        # Inequality Constraint
        # ======================================================================

        if self._trq_limit is None:
            if contact_list is not None:
                ineq_mat = np.concatenate((np.zeros(
                    (dim_cone_constraint, self._n_q_dot)), -uf_mat),
                                          axis=1)
                ineq_vec = -uf_vec
            else:
                ineq_mat = None
                ineq_vec = None

        else:
            if contact_list is not None:
                ineq_mat = np.concatenate(
                    (np.concatenate(
                        (np.zeros((dim_cone_constraint, self._n_q_dot)),
                         -np.dot(sa_ni_trc_bar_tr,
                                 np.dot(self._snf, self._mass_matrix)),
                         np.dot(sa_ni_trc_bar_tr,
                                np.dot(self._snf, self._mass_matrix))),
                        axis=0),
                     np.concatenate(
                         (-uf_mat,
                          np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                                 np.dot(contact_jacobian, ni).transpose()),
                          -np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                                  np.dot(contact_jacobian, ni).transpose())),
                         axis=0)),
                    axis=1)
                ineq_vec = np.concatenate(
                    (-uf_vec,
                     np.dot(
                         np.dot(sa_ni_trc_bar_tr, self._snf),
                         np.dot(ni.transpose(),
                                (self._coriolis + self._gravity))) +
                     np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) - self._trq_limit[:, 0],
                     -np.dot(
                         np.dot(sa_ni_trc_bar_tr, self._snf),
                         np.dot(ni.transpose(),
                                (self._coriolis + self._gravity))) -
                     np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) + self._trq_limit[:, 1]))

            else:
                ineq_mat = np.concatenate(
                    (-np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                             self._mass_matrix),
                     np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                            self._mass_matrix)),
                    axis=0)
                ineq_vec = np.concatenate(
                    (np.dot(
                        np.dot(sa_ni_trc_bar_tr, self._snf),
                        np.dot(ni.transpose(),
                               (self._coriolis + self._gravity))) +
                     np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) - self._trq_limit[:, 0],
                     -np.dot(
                         np.dot(sa_ni_trc_bar_tr, self._snf),
                         np.dot(ni.transpose(),
                                (self._coriolis + self._gravity))) -
                     np.dot(np.dot(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) + self._trq_limit[:, 1]))


        if self._trq_limit is not None:
            self.t_trq_limit = torch.from_numpy(self._trq_limit).expand(self.n_batch, -1)
        
        self.t_snf = torch.from_numpy(self._snf).expand(self.n_batch, -1, -1)

        if self._trq_limit is None:
            if contact_list is not None:
                t_ineq_mat = torch.cat((torch.zeros(self.n_batch, t_dim_cone_constraint, self._n_q_dot), 
                                        -t_uf_mat), dim=2)
                t_ineq_vec = -t_uf_vec
            else:
                t_ineq_mat = None
                t_ineq_vec = None

        else:
            if contact_list is not None:
                t_ineq_mat = torch.cat(
                    (torch.cat(
                        (torch.zeros((self.n_batch, t_dim_cone_constraint, self._n_q_dot)),
                         -torch.bmm(t_sa_ni_trc_bar_tr,
                                 torch.bmm(self.t_snf, self.t_mass_matrix)),
                         torch.bmm(t_sa_ni_trc_bar_tr,
                                torch.bmm(self.t_snf, self.t_mass_matrix))),
                        dim=1),
                     torch.cat(
                         (-t_uf_mat,
                          torch.bmm(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                                 torch.bmm(t_contact_jacobian, t_ni).transpose(1,2)),
                          -torch.bmm(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                                  torch.bmm(t_contact_jacobian, t_ni).transpose(1,2))),
                         dim=1)),
                    dim=2)
                t_ineq_vec = torch.cat(
                    (-t_uf_vec,
                     torch.matmul(
                         torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                         torch.matmul(t_ni.transpose(1,2),
                                (self.t_coriolis + self.t_gravity).unsqueeze(2))).squeeze() +
                     torch.matmul(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                            t_jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() - self.t_trq_limit[:, :, 0],
                     -torch.matmul(
                         torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                         torch.matmul(t_ni.transpose(1,2),
                                (self.t_coriolis + self.t_gravity).unsqueeze(2))).squeeze() -
                     torch.matmul(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                            t_jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() + self.t_trq_limit[:, :, 1]),
                    dim = 1)

            else:
                t_ineq_mat = torch.cat(
                    (-torch.bmm(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                             self.t_mass_matrix),
                     torch.bmm(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                            self.t_mass_matrix)),
                    dim=1)
                t_ineq_vec = torch.cat(
                    (torch.matmul(
                        torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                        torch.matmul(t_ni.transpose(1,2),
                               (self.t_coriolis + self.t_gravity).unsqueeze(2))).squeeze() +
                     torch.matmul(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                            t_jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() - self.t_trq_limit[:,:, 0],
                     -torch.matmul(
                         torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                         torch.matmul(t_ni.transpose(1,2),
                                (self.t_coriolis + self.t_gravity).unsqueeze(2))).squeeze() -
                     torch.matmul(torch.bmm(t_sa_ni_trc_bar_tr, self.t_snf),
                            t_jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() + self.t_trq_limit[:,:, 1]))
        """
        printeps("ineq_mat", ineq_mat - t_ineq_mat[0].numpy())
        printeps("ineq_vec", ineq_vec - t_ineq_vec[0].numpy())
        """
        # if verbose:
        # print("eq_mat")
        # print(eq_mat)
        # print("eq_vec")
        # print(eq_vec)

        # print("ineq_mat")
        # print(ineq_mat)
        # print("ineq_vec")
        # print(ineq_vec)
        #printvar("cost_mat", cost_mat)
        #printvar("cost_vec", cost_vec)
        #printvar("ineq_mat", ineq_mat)
        #printvar("ineq_vec", ineq_vec)
        #printvar("eq_mat", eq_mat)
        #printvar("eq_vec", eq_vec)
        self.cost_mat = cost_mat
        self.cost_vec = cost_vec
        self.ineq_mat = ineq_mat
        self.ineq_vec = ineq_vec
        self.eq_mat = eq_mat
        self.eq_vec = eq_vec

        #print(np.linalg.matrix_rank(eq_mat), eq_mat.shape)

        torch_cost_mat = torch.from_numpy(cost_mat).expand(2, -1, -1)
        torch_ineq_mat = torch.from_numpy(ineq_mat).expand(2, -1, -1)
        torch_eq_mat = torch.from_numpy(eq_mat).expand(2, -1, -1)

        torch_cost_vec = torch.from_numpy(cost_vec).expand(2, -1)
        torch_ineq_vec = torch.from_numpy(ineq_vec).expand(2, -1)
        torch_eq_vec = torch.from_numpy(eq_vec).expand(2, -1)

        #torch_sol = QPFunction(verbose = -1)(torch_cost_mat, torch_cost_vec, torch_ineq_mat, torch_ineq_vec, torch_eq_mat, torch_eq_vec)[0,:]



        printeps("ineq_mat", ineq_mat - t_ineq_mat[0].numpy())
        printeps("ineq_vec", ineq_vec - t_ineq_vec[0].numpy())
        printeps("cost_mat", cost_mat -t_cost_mat.numpy())
        printeps("cost_vec", cost_vec - t_cost_vec.numpy())
        printeps("eq_mat", eq_mat - t_eq_mat[0].numpy())
        printeps("eq_vec", eq_vec - t_eq_vec[0].numpy())

        
        cost_mat = t_cost_mat[0].numpy().astype(np.float64)
        cost_vec = t_cost_vec[0].numpy().astype(np.float64)
        eq_mat = t_eq_mat[0].numpy().astype(np.float64)
        ineq_mat = t_ineq_mat[0].numpy().astype(np.float64)
        eq_vec = t_eq_vec[0].numpy().astype(np.float64)
        ineq_vec = t_ineq_vec[0].numpy().astype(np.float64)

        print(cost_mat.shape, cost_vec.shape, ineq_mat.shape, ineq_mat.shape,
              eq_mat.shape, eq_vec.shape)
        """
        sol = solve_qp(cost_mat,
                       cost_vec,
                       ineq_mat,
                       ineq_vec,
                       eq_mat,
                       eq_vec,
                       solver="quadprog",
                       verbose=True)
        """
        #print("np solution", sol)
        #print("torch sol", torch_sol)
        """
        diffsol = sol-torch_sol.numpy()
        diffsol[diffsol <= 1e-4] = 0
        diffsol[diffsol > 1e-4] = 1
        #print(diffsol )
        if (np.linalg.norm(diffsol) > 1e-5):
            #time.sleep(20)
            print("/" *50)
            print("error")
            print("/"*50)
        sol = torch_sol.numpy()
        """
        sol = QPFunction(verbose = -1)(t_cost_mat, t_cost_vec, t_ineq_mat, t_ineq_vec, t_eq_mat, t_eq_vec)[0].numpy()



        if contact_list is not None:
            sol_q_ddot, sol_rf = sol[:self._n_q_dot], sol[self._n_q_dot:]
        else:
            sol_q_ddot, sol_rf = sol, None

        if contact_list is not None:
            joint_trq_cmd = np.dot(
                np.dot(sa_ni_trc_bar_tr, self._snf),
                np.dot(self._mass_matrix, sol_q_ddot) +
                np.dot(ni.transpose(), (self._coriolis + self._gravity)) -
                np.dot(np.dot(contact_jacobian, ni).transpose(), sol_rf))
        else:
            joint_trq_cmd = np.dot(
                np.dot(sa_ni_trc_bar_tr, self._snf),
                np.dot(self._mass_matrix, sol_q_ddot) +
                np.dot(ni, (self._coriolis + self._gravity)))

        joint_acc_cmd = np.dot(self._sa, sol_q_ddot)

        if verbose:
            # if True:
            print("joint_trq_cmd: ", joint_trq_cmd)
            print("sol_q_ddot: ", sol_q_ddot)
            print("sol_rf: ", sol_rf)

            # for i, task in enumerate(task_list):
            for task in [task_list[3], task_list[4]]:
                j = task.jacobian
                j_dot_q_dot = task.jacobian_dot_q_dot
                x_ddot = task.op_cmd
                print(task.target_id, " task")
                print("des x ddot: ", x_ddot)
                print("j*qddot_sol + Jdot*qdot: ",
                      np.dot(j, sol_q_ddot) + j_dot_q_dot)

        if self._b_data_save:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)
            self._data_saver.add('joint_acc_cmd', joint_acc_cmd)
            self._data_saver.add('rf_cmd', sol_rf)

        return joint_trq_cmd, joint_acc_cmd, sol_rf
