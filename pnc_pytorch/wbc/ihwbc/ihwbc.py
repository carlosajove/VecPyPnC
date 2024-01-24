import sys
import numpy as np
import torch

np.set_printoptions(precision=2, threshold=sys.maxsize)
from scipy.linalg import block_diag
from qpsolvers import solve_qp

from util import util
from pnc.data_saver import DataSaver


class IHWBC(object):
    """
    Implicit Hierarchy Whole Body Control
    ------------------
    Usage:
        update_setting --> solve
    """
    def __init__(self, sf, sa, sv, data_save=False):  #check for sf, sa, sv types must be torch tensors

        self._n_q_dot = sa.shape[1]   #Shape works equal in pytorch
        self._n_active = sa.shape[0]
        self._n_passive = sv.shape[0]

        self._sf = sf
        self._snf = torch.cat((torch.zeros(
            (self._n_active + self._n_passive, 6)),
                                    torch.eye(self._n_active + self._n_passive)),
                                   dim=1)
        self._sa = sa
        self._sv = sv

        self._trq_limit = None
        self._lambda_q_ddot = 0.
        self._lambda_rf = 0.
        self._w_rf = 0.
        self._w_hierarchy = 0.

        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()    #check data saver

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
        self._trq_limit = torch.clone(val).detach() 
        #This function is differentiable, 
        # so gradients will flow back from the result of this operation to input. 
        #To create a tensor without an autograd relationship to input see detach().

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

    """Carlos"""
    #the following must be torch tensors:
    #all inputs of update_setting

    def update_setting(self, mass_matrix, mass_matrix_inv, coriolis, gravity):
        self._mass_matrix = torch.clone(mass_matrix).detach()
        self._mass_matrix_inv = torch.clone(mass_matrix_inv).detach()
        self._coriolis = torch.clone(coriolis).detach()
        self._gravity = torch.clone(gravity).detach()

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

        """CARLOS"""
        #the following must be torch tensors:
        #ic.jacobian 
        #ic.jacobian_dot_q_dot
        #ji 
        #self._mass_matrix_inv
        #self._sa

        if len(internal_constraint_list) > 0:  
            ji = torch.cat(
                [ic.jacobian for ic in internal_constraint_list], axis=0)
            jidot_qdot = torch.cat(
                [ic.jacobian_dot_q_dot for ic in internal_constraint_list],
                axis=0)
            lmd = torch.linalg.pinv(
                torch.matmul(torch.matmul(ji, self._mass_matrix_inv), ji.t()))
            ji_bar = torch.matmul(torch.matmul(self._mass_matrix_inv, ji.t()), lmd)
            ni = torch.eye(self._n_q_dot) - torch.matmul(ji_bar, ji)
            jit_lmd_jidot_qdot = torch.squeeze(
                torch.matmul(torch.matmul(ji.t(), lmd), jidot_qdot))
            sa_ni_trc = torch.matmul(self._sa, ni)[:, 6:]
            """Carlos """
            #util.weighted_pinv
            sa_ni_trc_bar = util.weighted_pinv_pytorch(sa_ni_trc,
                                               self._mass_matrix_inv[6:, 6:])
            sa_ni_trc_bar_tr = sa_ni_trc_bar.t()
            b_internal_constraint = True
        else:
            ni = torch.eye(self._n_q_dot)
            jit_lmd_jidot_qdot = torch.zeros(self._n_q_dot)
            sa_ni_trc_bar = torch.eye(self._n_active)
            sa_ni_trc_bar_tr = sa_ni_trc_bar.t()
            b_internal_constraint = False

        # print("ni")
        # print(ni)
        # print("jit_lmd_jidot_qdot")
        # print(jit_lmd_jidot_qdot)
        # print("sa_ni_trc_bar_tr")
        # print(sa_ni_trc_bar_tr)
        # exit()

        # ======================================================================
        # Cost
        # ======================================================================
        cost_t_mat = torch.zeros((self._n_q_dot, self._n_q_dot))
        cost_t_vec = torch.zeros(self._n_q_dot)
        """CARLOS"""
        #the following must be torch tensors:
        #task.jacobian
        #task.jacobian_dot_q_dot
        #task.op_cmd
        #contact_list elements

        for i, task in enumerate(task_list):
            j = task.jacobian
            j_dot_q_dot = task.jacobian_dot_q_dot
            x_ddot = task.op_cmd
            if verbose:
                print("====================")
                print(task.target_id, " task")
                task.debug()

            cost_t_mat += self._w_hierarchy[i] * torch.matmul(j.t(), j)
            cost_t_vec += self._w_hierarchy[i] * torch.matmul(
                (j_dot_q_dot - x_ddot).t(), j)
        # cost_t_mat += self._lambda_q_ddot * np.eye(self._n_q_dot)
        cost_t_mat += self._lambda_q_ddot * self._mass_matrix

        """CARLOS"""
        #block_diag is a scipy function uses numpy 
        #Need to build the function myself in torch
        if contact_list is not None:

            """
            CARLOS
            BUILD OWN BLOCK DIAGRAM
            """
            uf_mat = np.array(
                block_diag(     
                    *[contact.cone_constraint_mat
                      for contact in contact_list])) 


            uf_vec = torch.cat(
                [contact.cone_constraint_vec for contact in contact_list])
            contact_jacobian = torch.cat(
                [contact.jacobian for contact in contact_list], axis=0)

            assert uf_mat.shape[0] == uf_vec.shape[0]
            assert uf_mat.shape[1] == contact_jacobian.shape[0]
            dim_cone_constraint, dim_contacts = uf_mat.shape

            cost_rf_mat = (self._lambda_rf + self._w_rf) * torch.eye(dim_contacts)
            if rf_des is None:
                rf_des = torch.zeros(dim_contacts)
            cost_rf_vec = -self._w_rf * torch.clone(rf_des).detach()

            """
            CARLOS
            BUILD OWN BLOCK DIAGRAM
            """
            cost_mat = np.array(block_diag(
                cost_t_mat, cost_rf_mat))  # (nqdot+nc, nqdot+nc)


            cost_vec = torch.cat([cost_t_vec, cost_rf_vec])  # (nqdot+nc,)

        else:
            dim_contacts = dim_cone_constraint = 0
            cost_mat = torch.clone(cost_t_mat).detach()
            cost_vec = torch.clone(cost_t_vec).detach()

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

        """CARLOS"""
        #the following must be torch tensors:
        #self._sf
        
        if contact_list is not None:
            eq_floating_mat = torch.cat(
                (torch.matmul(self._sf, self._mass_matrix),
                 -torch.matmul(self._sf,
                         torch.matmul(contact_jacobian, ni).t())),
                dim=1)  # (6, nqdot+nc)
            if b_internal_constraint:
                eq_int_mat = torch.cat(
                    (ji, torch.zeros(
                        (ji.shape[0], dim_contacts))), dim=1)  # (2, nqdot+nc)
                eq_int_vec = torch.zeros(ji.shape[0])
        else:
            eq_floating_mat = torch.matmul(self._sf, self._mass_matrix)
            if b_internal_constraint:
                eq_int_mat = torch.clone(ji).detach()
                eq_int_vec = torch.zeros(ji.shape[0])
        eq_floating_vec = -torch.matmul(
            self._sf, torch.matmul(ni.t(), (self._coriolis + self._gravity)))

        if b_internal_constraint:
            eq_mat = torch.cat((eq_floating_mat, eq_int_mat), dim=0)
            eq_vec = torch.cat((eq_floating_vec, eq_int_vec), dim=0)
        else:
            eq_mat = torch.clone(eq_floating_mat).detach()
            eq_vec = torch.clone(eq_floating_vec).detach()

        # ======================================================================
        # Inequality Constraint
        # ======================================================================

        if self._trq_limit is None:
            if contact_list is not None:
                ineq_mat = torch.cat((torch.zeros(
                    (dim_cone_constraint, self._n_q_dot)), -uf_mat),
                                          dim=1)
                ineq_vec = -uf_vec
            else:
                ineq_mat = None
                ineq_vec = None

        else:
            if contact_list is not None:
                ineq_mat = torch.cat(
                    (torch.cat(
                        (torch.zeros((dim_cone_constraint, self._n_q_dot)),
                         -torch.matmul(sa_ni_trc_bar_tr,
                                 torch.matmul(self._snf, self._mass_matrix)),
                         torch.matmul(sa_ni_trc_bar_tr,
                                torch.matmul(self._snf, self._mass_matrix))),
                        dim=0),
                     torch.cat(
                         (-uf_mat,
                          torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                                 torch.matmul(contact_jacobian, ni).t()),
                          -torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                                  torch.matmul(contact_jacobian, ni).t())),
                         dim=0)),
                    dim=1)
                ineq_vec = torch.cat(
                    (-uf_vec,
                     torch.matmul(
                         torch.matmul(sa_ni_trc_bar_tr, self._snf),
                         torch.matmul(ni.t(),
                                (self._coriolis + self._gravity))) +
                     torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) - self._trq_limit[:, 0],
                     -torch.matmul(
                         torch.matmul(sa_ni_trc_bar_tr, self._snf),
                         torch.matmul(ni.t(),
                                (self._coriolis + self._gravity))) -
                     torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) + self._trq_limit[:, 1]))

            else:
                ineq_mat = torch.cat(
                    (-torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                             self._mass_matrix),
                     torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                            self._mass_matrix)),
                    dim=0)
                ineq_vec = torch.cat(
                    (torch.matmul(
                        torch.matmul(sa_ni_trc_bar_tr, self._snf),
                        torch.matmul(ni.t(),
                               (self._coriolis + self._gravity))) +
                     torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) - self._trq_limit[:, 0],
                     -torch.matmul(
                         torch.matmul(sa_ni_trc_bar_tr, self._snf),
                         torch.matmul(ni.t(),
                                (self._coriolis + self._gravity))) -
                     torch.matmul(torch.matmul(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot) + self._trq_limit[:, 1]))

        # if verbose:
        # print("eq_mat")
        # print(eq_mat)
        # print("eq_vec")
        # print(eq_vec)

        # print("ineq_mat")
        # print(ineq_mat)
        # print("ineq_vec")
        # print(ineq_vec)
        """CARLOS """
        #MUST CHANGE THE SOLVER
        sol = solve_qp(cost_mat,
                       cost_vec,
                       ineq_mat,
                       ineq_vec,
                       eq_mat,
                       eq_vec,
                       solver="quadprog",
                       verbose=True)

        if contact_list is not None:
            sol_q_ddot, sol_rf = sol[:self._n_q_dot], sol[self._n_q_dot:]
        else:
            sol_q_ddot, sol_rf = sol, None

        if contact_list is not None:
            joint_trq_cmd = torch.matmul(
                torch.matmul(sa_ni_trc_bar_tr, self._snf),
                torch.matmul(self._mass_matrix, sol_q_ddot) +
                torch.matmul(ni.t(), (self._coriolis + self._gravity)) -
                torch.matmul(torch.matmul(contact_jacobian, ni).t(), sol_rf))
        else:
            joint_trq_cmd = torch.matmul(
                torch.matmul(sa_ni_trc_bar_tr, self._snf),
                torch.matmul(self._mass_matrix, sol_q_ddot) +
                torch.matmul(ni, (self._coriolis + self._gravity)))

        joint_acc_cmd = torch.matmul(self._sa, sol_q_ddot)

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
                      torch.matmul(j, sol_q_ddot) + j_dot_q_dot)

        if self._b_data_save:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)
            self._data_saver.add('joint_acc_cmd', joint_acc_cmd)
            self._data_saver.add('rf_cmd', sol_rf)

        return joint_trq_cmd, joint_acc_cmd, sol_rf