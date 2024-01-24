import torch
from mpc import mpc


class alip_mpc():
    def __init__(self, Nsteps, Nintervals, Qrunning):
        self.g = 9.806
        self.N_intervals = Nintervals
        self.N_steps = Nsteps
        self.Q_running = Qrunning
        #dimensions
        self.n_state = 4
        self.n_ufp = 2



    def solve_left_stance(self, indata): #indata.st_leg = 


        x_sol, ufp_sol, objs_lqr = (n.s)



    def solve_right_stance(self, indata):
        a


    def x_dynamics(self, s, x):
        a
        
    def cost_function(self):
        a
    
    def x_bound(x, mu, kx, ky, zH, ufp_max):  #this condition must be placed on the x_dynamics function
        #slip:
        #xc_slip_limit = (mu - kx)*zH / (1 + kx^2);
        #yc_slip_limit = (mu - ky)*zH / (1 + ky^2);
        mut = torch.tensor([mu,mu])
        k = torch.tensor([kx,ky])
        slip_limit = torch.div(torch.torch.mul(torch.sub(mut,k), zH), torch.add(torch.mul(k,k), 1))

        #mechanical
        mech_limit = torch.tensor([ufp_max, float('inf')])

        x_upper = torch.min(slip_limit, mech_limit)
        x_lower = torch.mul(x_upper, -1)

        return x_upper, x_lower

    def u_bound_left_stance(ufp_max, ufp_min, N_steps):

    
    

    






