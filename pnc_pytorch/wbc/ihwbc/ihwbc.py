import sys
import numpy as np
import torch

np.set_printoptions(precision=2, threshold=sys.maxsize)
from scipy.linalg import block_diag
from util import util
from pnc_pytorch.data_saver import DataSaver
from qpsolvers import solve_qp

from pnc_pytorch.wbc.ihwbc.qpth.qp import QPFunction   #for now like this for testing 
                                                       #afterwards should put in conda 

def printvar(a, b):
    print(a, "\n", b, " shape" , b.shape, " | type", b.dtype, "\n")

def is_psd2(mat):
    is_symmetric = (mat == mat.T).all()
    eigenvalues_non_negative = (torch.linalg.eigvals(mat).real >= 0).all()
    
    if not (is_symmetric and eigenvalues_non_negative):
        print("The matrix is not positive semidefinite.")
        if not is_symmetric:
            print("Reason: The matrix is not symmetric (mat != mat.T).")
        if not eigenvalues_non_negative:
            print("Reason: The eigenvalues of the matrix are not all non-negative.")
            print("Eigenvalues: ", torch.linalg.eigvals(mat).real)
            
    return is_symmetric and eigenvalues_non_negative

def ensure_psd_add_Id(matrix, epsilon):
    m = torch.eye(matrix.shape[1], matrix.shape[2]).repeat(matrix.shape[0], 1, 1)
    return matrix + epsilon*m


class IHWBC(object):
    """
    Implicit Hierarchy Whole Body Control
    ------------------
    Usage:
        update_setting --> solve
    """
    def __init__(self, sf, sa, sv, n_batch, data_save=False):  
        #for sf, sa, sv types must be batched torch tensors: 
        self.n_batch = n_batch
        self._n_q_dot = sa.shape[2]   
        self._n_active = sa.shape[1]
        self._n_passive = sv.shape[1]

        self._sf = sf
        self._snf = torch.cat((torch.zeros(
            self.n_batch, self._n_active + self._n_passive, 6, dtype=torch.double),
            torch.eye(self._n_active + self._n_passive).unsqueeze(0).repeat(self.n_batch,1,1)),
                                   dim=2)
        self._sa = sa
        self._sv = sv

        self._trq_limit = None       #torch.tensor([n_batch, n_active])
        self._lambda_q_ddot = 0.     #just dim 1
        self._lambda_rf = 0.    #must be dim 1
        self._w_rf = 0.         #must be dim 1
        self._w_hierarchy = 0.  #must be [n_batch , #tasks]

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
        assert val.shape[1] == self._n_active
        assert val.shape[0] == self.n_batch
        self._trq_limit = torch.clone(val).detach() 

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

    #input size : [n_batch, ...]
    def update_setting(self, mass_matrix, mass_matrix_inv, coriolis, gravity):
        self._mass_matrix = torch.clone(mass_matrix).detach()
        self._mass_matrix_inv = torch.clone(mass_matrix_inv).detach()
        self._coriolis = torch.clone(coriolis).detach() #dim: nbatch x vector
        self._gravity = torch.clone(gravity).detach() #dim: nbatch x vector

    def solve(self,
              task_list,
              contact_list,
              internal_constraint_list,
              rf_des=None,
              verbose=False):
        """
        Parameters
        ----------
        task_list (list of batched-Task): was list of Task
            Task list
        contact_list (list of batched-Contact):  was list of Contact
            Contact list
        internal_constraint_list (list of batched-InternalConstraint):
            Internal constraint list
        rf_des (torch.tensor([n_batch, ...])):
            Reaction force desired
        verbose (bool):
            Printing option

        Returns
        -------
        joint_trq_cmd (torch.tensor([n_batch, ...])):
            Joint trq cmd
        joint_acc_cmd (torch.tensor([n_batch, ...])):
            Joint acc cmd
        sol_rf (torch.tensor([n_batch, ...])):
            Reaction force
        """

        # ======================================================================
        # Internal Constraint
        #   Set ni, jit_lmd_jidot_qdot, sa_ni_trc_bar_tr, and b_internal_constraint
        # ======================================================================

        """CARLOS"""
        #the following must be n_batched torch tensors:
        #ic.jacobian 
        #ic.jacobian_dot_q_dot --->batched vector
        #ji 
        #self._mass_matrix_inv
        #self._sa

        if len(internal_constraint_list) > 0:  
            ji = torch.cat(
                [ic.jacobian for ic in internal_constraint_list], axis=1)     #ic.jacobian: [n_batch, etc]
            jidot_qdot = torch.cat(
                [ic.jacobian_dot_q_dot for ic in internal_constraint_list],
                axis=1)

            lmd = torch.linalg.pinv(
                torch.bmm(torch.bmm(ji, self._mass_matrix_inv), ji.transpose(1,2)))

            ji_bar = torch.bmm(torch.bmm(self._mass_matrix_inv, ji.transpose(1,2)), lmd)
            ni = torch.eye(self._n_q_dot).unsqueeze(0).repeat(self.n_batch, 1, 1) - torch.bmm(ji_bar, ji)

            jit_lmd_jidot_qdot = torch.squeeze(
                torch.matmul(torch.bmm(ji.transpose(1,2), lmd), jidot_qdot.unsqueeze(2)))
            sa_ni_trc = torch.bmm(self._sa, ni)[:, :, 6:]

            #TODO: check util funciton
            sa_ni_trc_bar = util.weighted_pinv_pytorch(sa_ni_trc,
                                            self._mass_matrix_inv[:, 6:, 6:])
            sa_ni_trc_bar_tr = sa_ni_trc_bar.transpose(1, 2)
            b_internal_constraint = True
        else:
            ni = torch.eye(self._n_q_dot).unsqueeze(0).repeat(self.n_batch, 1, 1)
            jit_lmd_jidot_qdot = torch.zeros(self._n_q_dot).unsqueeze(0).repeat(self.n_batch, 1)
            sa_ni_trc_bar = torch.eye(self._n_active).unsqueeze(0).repeat(self._n_active, 1, 1)
            sa_ni_trc_bar_tr = sa_ni_trc_bar.transpose(1,2)
            b_internal_constraint = False

        """
        printvar("ni", ni[0])
        printvar("jit_lmd_jidot_qdot", jit_lmd_jidot_qdot[0])
        printvar("sa_ni_trc_bar_tr", sa_ni_trc_bar_tr[0])
        # exit()
        """

        # ======================================================================
        # Cost
        # ======================================================================
        cost_t_mat = torch.zeros(self.n_batch, self._n_q_dot, self._n_q_dot).to(torch.float64)
        cost_t_vec = torch.zeros(self.n_batch, self._n_q_dot).to(torch.float64)

        #the following must be batched torch tensors:
        #task.jacobian
        #task.jacobian_dot_q_dot
        #task.op_cmd
        #self._w_hierarchy
        #task.op_cmd
        for i, task in enumerate(task_list):
            j = task.jacobian
            j_dot_q_dot = task.jacobian_dot_q_dot
            x_ddot = task.op_cmd
            if verbose:
                print("====================")
                print(task.target_id, " task")
                task.debug()
            """
            print("IHWBC")
            print("jacobian rank", torch.linalg.matrix_rank(j), j.shape)
            printvar("w", self._w_hierarchy)
            printvar("jdotqdot", j_dot_q_dot)
            printvar("xddot", x_ddot)
            printvar("jacobian", j)
            """
            jTj_psd = ensure_psd_add_Id(torch.bmm(j.transpose(1,2).to(torch.float64), j.to(torch.float64)), 1e-9)

            cost_t_mat += self._w_hierarchy[:,i].unsqueeze(1).unsqueeze(1).to(torch.float64) * jTj_psd.to(torch.float64)

            cost_t_vec += self._w_hierarchy[:,i].unsqueeze(1).to(torch.float64) * torch.matmul(
                (j_dot_q_dot.to(torch.float64) - x_ddot.to(torch.float64)).unsqueeze(1), j.to(torch.float64)).squeeze().to(torch.float64)

        # cost_t_mat += self._lambda_q_ddot * np.eye(self._n_q_dot)
        #TODO: check why uses mass matrix
        #printvar("cost_t_mat", cost_t_mat[0])
        cost_t_mat += self._lambda_q_ddot * self._mass_matrix.to(torch.float64)

        #printvar("cost_t_mat 2 ", cost_t_mat[0])
        #printvar("cost t vec ", cost_t_vec[0])

        #contact.cone_contraint_vec: torch.tensor([n_batch, 6])
        #contact.constraint_mat: torch.tensor([n_batch, 6, dim_contact)
        #contact.jacobian: torch.tensor([n_batch, dim_contact, robot.n_q])
        if contact_list is not None:   

            """
            CARLOS: For now use this to make batched block diagram
            """
            uf_mat_list = []
            for i in range(self.n_batch):
                uf_mat_list.append(torch.block_diag(*[contact.cone_constraint_mat[i]
                    for contact in contact_list]))
            uf_mat = torch.stack([*uf_mat_list], axis = 0)
            """end batched """

            uf_vec = torch.cat(
                [contact.cone_constraint_vec for contact in contact_list], axis = 1)
            contact_jacobian = torch.cat(
                [contact.jacobian for contact in contact_list], axis=1)
            assert uf_mat.shape[1] == uf_vec.shape[1]
            assert uf_mat.shape[2] == contact_jacobian.shape[1]

            dim_cone_constraint = uf_mat.shape[1]
            dim_contacts = uf_mat.shape[2]

            cost_rf_mat = (self._lambda_rf + self._w_rf) * torch.eye(dim_contacts)
            #doesn't need to be batched, since params don't change between sims

            #TODO: make sure that i fwe don't have desired reaction forces, just put 0 instead
            if rf_des is None: #TODO: Make sure 
                rf_des = torch.zeros(self.n_batch, dim_contacts)
            
            cost_rf_vec = -self._w_rf * torch.clone(rf_des).detach()#rf_des torch.tensor([n_batch, ...])

            cost_mat_list = []
            for i in range(self.n_batch):
                cost_mat_list.append(torch.block_diag(cost_t_mat[i], cost_rf_mat)) # (nqdot+nc, nqdot+nc)
            cost_mat = torch.stack([*cost_mat_list], axis = 0)
            


            cost_vec = torch.cat([cost_t_vec, cost_rf_vec], axis=1)  # (nqdot+nc,)

        else:
            dim_contacts = dim_cone_constraint = 0
            cost_mat = torch.clone(cost_t_mat).detach()
            cost_vec = torch.clone(cost_t_vec).detach()

        #printvar("cost_t_mat 3", cost_t_mat[0])

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
        #the following must be batched torch tensors:
        #self._sf
        #self._mass_matrix
        #contact_jacobian
        #ni
        #ji 
        #TODO: check if b_internal_constraint depends on batch, right now assume same for all
        #TODO: check why ni
        #print("rank ni", torch.linalg.matrix_rank(ni), ni.shape)
        if contact_list is not None:
            eq_floating_mat = torch.cat(
                (torch.bmm(self._sf, self._mass_matrix),
                 -torch.bmm(self._sf,torch.bmm(contact_jacobian, ni).transpose(1,2))),
                dim=2)  # (6, nqdot+nc)
            if b_internal_constraint:
                eq_int_mat = torch.cat(
                    (ji, torch.zeros(self.n_batch, ji.shape[1], dim_contacts)), dim=2)  # (2, nqdot+nc)
                eq_int_vec = torch.zeros(self.n_batch, ji.shape[1])
        else:
            eq_floating_mat = torch.bmm(self._sf, self._mass_matrix)
            if b_internal_constraint:
                eq_int_mat = torch.clone(ji).detach()
                eq_int_vec = torch.zeros(self.n_batch, ji.shape[1])
        #printvar("ni^T", ni.transpose(1,2))
        #printvar("coriolis", self._coriolis)
        #printvar("grav", self._gravity)
        #printvar("unsqueeze", (self._coriolis + self._gravity).unsqueeze(2))
        #print(torch.matmul(ni.transpose(1,2), (self._coriolis + self._gravity).unsqueeze(2)))
        eq_floating_vec = -torch.matmul(
            self._sf, torch.matmul(ni.transpose(1,2), (self._coriolis + self._gravity).unsqueeze(2))).squeeze(2)

        #printvar("eq_floating_mat", eq_floating_mat)
        #printvar("eq_int_mat", eq_int_mat)
        #printvar("eq_floaating_vec", eq_floating_vec)
        #printvar("eq_int_vec", eq_int_vec)
        if b_internal_constraint:
            eq_mat = torch.cat((eq_floating_mat, eq_int_mat), dim=1)
            eq_vec = torch.cat((eq_floating_vec, eq_int_vec), dim=1)
        else:
            eq_mat = torch.clone(eq_floating_mat).detach()
            eq_vec = torch.clone(eq_floating_vec).detach()

        # ======================================================================
        # Inequality Constraint
        # ======================================================================

        if self._trq_limit is None:
            if contact_list is not None:
                ineq_mat = torch.cat((torch.zeros(self.n_batch, dim_cone_constraint, self._n_q_dot), 
                                        -uf_mat), dim=2)
                ineq_vec = -uf_vec
            else:
                ineq_mat = None
                ineq_vec = None

        else:
            print("HELLO")
            if contact_list is not None:
                ineq_mat = torch.cat(
                    (torch.cat(
                        (torch.zeros((self.n_batch, dim_cone_constraint, self._n_q_dot)),
                         -torch.bmm(sa_ni_trc_bar_tr,
                                 torch.bmm(self._snf, self._mass_matrix)),
                         torch.bmm(sa_ni_trc_bar_tr,
                                torch.bmm(self._snf, self._mass_matrix))),
                        dim=1),
                     torch.cat(
                         (-uf_mat,
                          torch.bmm(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                                 torch.bmm(contact_jacobian, ni).transpose(1,2)),
                          -torch.bmm(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                                  torch.bmm(contact_jacobian, ni).transpose(1,2))),
                         dim=1)),
                    dim=2)
                ineq_vec = torch.cat(
                    (-uf_vec,
                     torch.matmul(
                         torch.bmm(sa_ni_trc_bar_tr, self._snf),
                         torch.matmul(ni.transpose(1,2),
                                (self._coriolis + self._gravity).unsqueeze(2))).squeeze() +
                     torch.matmul(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() - self._trq_limit[:, :, 0],
                     -torch.matmul(
                         torch.bmm(sa_ni_trc_bar_tr, self._snf),
                         torch.matmul(ni.transpose(1,2),
                                (self._coriolis + self._gravity).unsqueeze(2))).squeeze() -
                     torch.matmul(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() + self._trq_limit[:, :, 1]),
                    dim = 1)

            else:
                ineq_mat = torch.cat(
                    (-torch.bmm(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                             self._mass_matrix),
                     torch.bmm(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                            self._mass_matrix)),
                    dim=1)
                ineq_vec = torch.cat(
                    (torch.matmul(
                        torch.bmm(sa_ni_trc_bar_tr, self._snf),
                        torch.matmul(ni.transpose(1,2),
                               (self._coriolis + self._gravity).unsqueeze(2))).squeeze() +
                     torch.matmul(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() - self._trq_limit[:,:, 0],
                     -torch.matmul(
                         torch.bmm(sa_ni_trc_bar_tr, self._snf),
                         torch.matmul(ni.transpose(1,2),
                                (self._coriolis + self._gravity).unsqueeze(2))).squeeze() -
                     torch.matmul(torch.bmm(sa_ni_trc_bar_tr, self._snf),
                            jit_lmd_jidot_qdot.unsqueeze(2)).squeeze() + self._trq_limit[:,:, 1]))

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
        #torch.set_printoptions(threshold=10000)
        #printvar("cost_mat", cost_mat[0,:,:])
        #print(torch.linalg.matrix_rank(cost_mat[0, :, :]))
        #printvar("cost_vec", cost_vec[0,:])
        #printvar("ineq_mat", ineq_mat[0,:,:])
        #printvar("ineq_vec", ineq_vec[0,:])
        #printvar("eq_mat", eq_mat[0,:,:])
        #printvar("eq_vec", eq_vec[0,:])
        self.cost_mat = cost_mat[0,:,:]
        self.cost_vec = cost_vec[0,:]
        self.ineq_mat = ineq_mat[0,:,:]
        self.ineq_vec = ineq_vec[0,:]
        self.eq_mat = eq_mat[0,:,:]
        self.eq_vec = eq_vec[0,:]

        #exit()
        eps = 1e-3
        #cost_mat += eps*torch.eye(cost_mat.shape[1], cost_mat.shape[2]).unsqueeze(0).repeat(self.n_batch, 1, 1)
        #print("cost_mat", cost_mat)
        #print("eq_mat", eq_mat[0,:,:])
        #print("torch", torch.linalg.matrix_rank(eq_mat[0,:,:]), eq_mat[0,:,:].shape)
        #eq_mat += eps*torch.eye(eq_mat.shape[1], eq_mat.shape[2]).unsqueeze(0).repeat(self.n_batch, 1, 1)

        #TODO: if nan's on solution check cost_mat and eq_mat
        #sol = QPFunction(verbose = -1)(cost_mat.float(), cost_vec.float(), ineq_mat, ineq_vec, eq_mat, eq_vec)

        cost_mat = cost_mat.double()
        cost_vec = cost_vec.double()
        ineq_mat = ineq_mat.double()
        ineq_vec = ineq_vec.double()
        eq_mat = eq_mat.double()
        eq_vec = eq_vec.double()
        """
        print("cost psd:", is_psd2(cost_mat[0]))
        print(cost_mat[0].numpy(), 
              cost_vec[0].numpy(), 
              ineq_mat[0].numpy(), 
              ineq_vec[0].numpy(),
              eq_mat[0].numpy(), 
              eq_vec[0].numpy())
        """
        """
        sol = solve_qp(cost_mat[0].numpy(), 
                       cost_vec[0].numpy(), 
                       ineq_mat[0].numpy(), 
                       ineq_vec[0].numpy(), 
                       eq_mat[0].numpy(), 
                       eq_vec[0].numpy(),
                       solver = "quadprog", verbose = False)
        
        sol = torch.from_numpy(sol).expand(self.n_batch, -1).float()
        """
        """
        print("cost psd:", is_psd2(cost_mat[0]))
        print(torch.isnan(cost_mat).any().item())  
        print(torch.isnan(cost_vec).any().item())  
        print(torch.isnan(eq_mat).any().item())  
        print(torch.isnan(ineq_mat).any().item())  
        print(torch.isnan(eq_vec).any().item())  
        print("CHECK")
        print(torch.isnan(ineq_vec).any().item())  
        print(ineq_vec)
        """
        sol = QPFunction(verbose = -1)(cost_mat, cost_vec, ineq_mat, ineq_vec, eq_mat, eq_vec)


        self.sol = sol
        #sol -> troch.tensor([n_batch, vecsize])
        #printvar("solution", sol)


        if contact_list is not None:
            sol_q_ddot, sol_rf = sol[:, :self._n_q_dot], sol[:, self._n_q_dot:]
        else:
            sol_q_ddot, sol_rf = sol, None

        if contact_list is not None:
            #TODO: check if i should squeeze to '+'
            joint_trq_cmd = torch.matmul(
                torch.bmm(sa_ni_trc_bar_tr, self._snf),
                torch.matmul(self._mass_matrix, sol_q_ddot.unsqueeze(2)) +
                torch.matmul(ni.transpose(1,2), (self._coriolis + self._gravity).unsqueeze(2)) -
                torch.matmul(torch.bmm(contact_jacobian, ni).transpose(1,2), sol_rf.unsqueeze(2))).squeeze(2)
        else:
            joint_trq_cmd = torch.matmul(
                torch.bmm(sa_ni_trc_bar_tr, self._snf),
                torch.matmul(self._mass_matrix, sol_q_ddot.unsqueeze(2)) +
                torch.matmul(ni, (self._coriolis + self._gravity).unsqueeze(2))).squeeze(2)

        joint_acc_cmd = torch.matmul(self._sa, sol_q_ddot.unsqueeze(2)).squeeze(2)

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
                      torch.matmul(j, sol_q_ddot.unsqueeze(2)).squeeze() + j_dot_q_dot)

        if self._b_data_save:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)
            self._data_saver.add('joint_acc_cmd', joint_acc_cmd)
            self._data_saver.add('rf_cmd', sol_rf)
        
        """
        print("IHWBC")
        print(joint_trq_cmd, joint_acc_cmd, sol_rf)
        """
        return joint_trq_cmd, joint_acc_cmd, sol_rf
