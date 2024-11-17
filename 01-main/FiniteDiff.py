from support import *

import numpy as np
from scipy import sparse
import sympy as sp
import matplotlib.pyplot as plt

x,t,c,L = sp.symbols('x,t,c,L')

class FDSolver:
    """Template class for the finite difference solvers"""

    def __init__(self,
                 N,
                 domain=[0,1],
                 cfl=1,
                 u0=None):
        
        self.N = N
        self.L0 = domain[0]
        self.LN = domain[1]
        self.cfl = cfl
        self.u0 = u0
        self.dx = domain[1]/N
        #self.dy = domain[1][1]/N[1]
        self.x = np.linspace(domain[0],domain[1],N+1)
        #self.y = np.linspace(domain[1][0],domain[1][1],N+1)

    @property
    def dt(self):
        raise RuntimeError

    def D2(self,bc):
        raise RuntimeError
    
    def apply_bcs(self, bc, u=None):
        raise RuntimeError
    
    def solver(self,Nt,cfl=None, bc=0, ic=0):
        raise RuntimeError
    
    def u_exact(self):
        raise RuntimeError
    

class Diffusion1DSolver(FDSolver):

    def __init__(self, N, domain, cfl, u0, D):
        super().__init__(N, domain, cfl, u0)
        self.Df = D
        self.unp1 = np.zeros(N+1)
        self.un = np.zeros(N+1)

    @property
    def dt(self):
        return self.cfl*self.dx**2/self.Df 
    
    def D2(self,bc):
        """ Defines the second order differentiation matrix and sets boundary
            conditions for Neumann boundary conditions by modifying D².
            Returns a unscaled version of D². 
        """
        
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        bcl,bcr = bc['left'], bc['right']

        ## Left boundary
        if bcl == 0:
            pass#D[0,:] = ...
        elif bcl == 1:
            D[0,:2] = -2, 2

        ## Right boundary
        if bcr == 0:
            pass #D[-1,:] = ...
        elif bcr == 1:
            D[-1,-2:] = 2, -2

        return D


    def apply_bcs(self, tn, bc, u=None):
        
        u = u if u is not None else self.unp1
        #if u is not None:
        #    un = self.un
        bcl,bcr = bc['left'], bc['right']
        
        ue_dx = sp.diff(self.ue, x, 1)

        ## Left boundary
        if bcl == 0:
            #print('here')
            #print(self.ue.subs({L: self.LN, x: self.L0, t: tn}))
            u[0] = self.ue.subs({L: self.LN, x: self.L0, t: tn})
        elif bcl == 1:
            u[0] = self.un[0] + self.cfl*(2*self.un[1]-2*self.un[0]) - ...
            ... - (self.cfl/self.dx)*ue_dx.subs({L: self.LN, x: self.L0, t: tn}) + ... 
            + self.dt*self.ue.subs({L: self.LN, x: self.L0, t: tn})

        ## Right boundary
        if bcr == 0:
            u[-1] = self.ue.subs({L: self.LN, x: self.LN, t: tn})
        elif bcr == 1:
            u[-1] = self.un[-1] + self.cfl*(2*self.un[-2]-2*self.un[-2]) - ...
            ... - (self.cfl/self.dx)*ue_dx.subs({L: self.LN, x: self.LN, t: tn}) + ...
            ... + self.dt*self.ue.subs({L: self.LN, x: self.LN, t: tn})

        return u
    
    def solver(self, time, cfl=None, bc={'left': 0, 'right': 0},ic=0):

        self.ue = self.u_exact()

        ## Initialization of D²-matrix
        D = self.D2(bc)

        ## Setting Courant number, time-step and initial solution
        #self.cfl = C = self.cfl if cfl is None else self.c*self.dt/self.dx
        self.cfl = C = self.Df*self.dt/self.dx**2
        print(C)
        dt = self.dt
        Nt = int(time/dt)

        ## Setting initical condition for u_n-array
        u0 = sp.lambdify(x, self.u0.subs({L: self.LN, t: 0}))
        self.un[:] = u0(self.x)

        self.sol_data = {0: self.un.copy()}

        for n in range(1,Nt+1):
            self.unp1[:] = self.un + C* (D @ self.un)
            self.unp1 = self.apply_bcs(tn=n*self.dt, bc=bc, u=self.unp1)
            
            ## Assigning updated solution for next timestep
            self.un[:] = self.unp1

            self.sol_data[n] = self.unp1.copy()

    def plot_result(self,time):

        Nt = int(time/self.dt)
        t_n = np.linspace(0,time,Nt+1)
        xx,tt = np.meshgrid(problem.x,t_n,indexing='ij')

        analytic_sol = self.u_exact()

        self.fd_solution = fd_sol = np.zeros((self.N+1,Nt+1))
        analytic = np.zeros_like(fd_sol)
        for n in range(len(t_n)):

            self.fd_solution[:,n] = self.sol_data[n]
            analytic_sol_tn = sp.lambdify(x, analytic_sol.subs({L: self.LN, t: t_n[n]}))
            analytic[:,n] = analytic_sol_tn(self.x)

        fd_sol = self.fd_solution
        self.abs_diff = np.abs(fd_sol - analytic)

        plot2D(tt,xx,self.fd_solution,labels=['FD solution','t','x','u(x,t)'])
        plot2D(tt,xx,analytic,labels=['Analytic solution','t','x','u(x,t)'])
        plot2D(tt,xx,self.abs_diff,labels=['Difference','t','x','u(x,t)'])


    def u_exact(self, amp=1.):
        Df = self.Df
        return amp*sp.exp(-sp.pi**2 * Df * t/(L**2)) * sp.sin(sp.pi * x/L)
    

class Wave1DSolver(FDSolver):

    def __init__(self, N, domain, cfl, u0, c):
        super().__init__(N, domain, cfl, u0)
        self.c = c
        self.unp1 = np.zeros(N+1)
        self.un = np.zeros(N+1)
        self.unm1 = np.zeros(N+1)

    @property
    def dt(self):
        return self.cfl*self.dx/self.c
    

    def D2(self, bc):
        ## Second order differentiation matrix, D²
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        bcl,bcr = bc['left'], bc['right']

        ## Boundary condtions that can be set directly in the D²-matrix
        # Left boundary
        if bcl == 0:    # Dirichlet condition
            #D[0,:] = 0
            pass
        elif bcl == 1:  # Neumann condition
            D[0,:2] = -2, 2

        # Right boundary
        if bcr == 0:    # Dirichlet condition
            #D[-1,:] = 0
            pass
        elif bcr == 1:  # Neumann condition
            D[-1,-2:] = 2,-2

        return D

    def apply_bcs(self, tn, bc, u=None):
        u = u if u is not None else self.unp1
        bcl, bcr = bc['left'], bc['right']
        ue_dx = sp.diff(self.ue, x, 1)
        ## Boundary conditions set for arb. RHS, f(x,t) for x = {0, L}
        # Left boundary
        if bcl == 0:    # Dirichlet condition, left, set in D²-matrix
            u[0] = self.ue.subs({L: self.LN, c: self.c, x: self.L0, t: tn})
        elif bcl == 1:  # Neumann condition, left, set in D²-matrix
            u[0] = 2*self.un[0] - self.unm1[0] + self.cfl**2*(2*self.un[1] - self.un[0]) - ...
            ... - 2*self.cfl**2*self.dx*ue_dx.subs({L: self.LN, c: self.c, x: self.L0, t: tn}) + ...
            ... + self.dt**2*self.ue.subs({L: self.LN, c: self.c, x: self.L0, t: tn})

        # Right boundary
        if bcr == 0: # Dirichlet condition, right, set in D²-matrix
            u[-1] = self.ue.subs({L: self.LN, c: self.c, x: self.LN, t: tn})
        elif bcr == 1: # Neumann condition, right, set in D²-matrix
            u[-1] = 2*self.un[-1] - self.unm1[-1] + self.cfl**2*(2*self.un[-2] - self.un[-1]) + ...
            ... + 2*self.cfl**2*self.dx*ue_dx.subs({L: self.LN, c: self.c, x: self.LN, t: tn}) + ...
            ... + self.dt**2*self.ue.subs({L: self.LN, c: self.c, x: self.LN, t: tn})

    def solver(self, time, cfl=None, bc=0, ic=0):

        self.ue = self.u_exact()

        ## Initialization of D²-matrix
        D = self.D2(bc)

        ## Setting Courant number, time-step and initial solution
        #self.cfl = C = self.cfl if cfl is None else self.c*self.dt/self.dx
        self.cfl = C = self.c*self.dt/self.dx
        print(C)
        dt = self.dt
        Nt = int(time/dt)


        ## First step. Set u_nm1-array
        u0 = sp.lambdify(x, self.u0.subs({L: self.LN, c: self.c, t: 0}))
        self.unm1[:] = u0(self.x)

        ## Time-series storage
        self.sol_data = {0: self.unm1.copy()}

        ## Setting initial conditions for u_n-array,
        #  either with specified function or u_t = 0
        if ic == 0: # use sympy function
            u0 = sp.lambdify(x, self.u0.subs({L: self.LN, c: self.c, t: dt}))
            self.un[:] = u0(self.x)
        else: 
            self.un[:] = self.unm1 + 0.5*C**2* (D @ self.unm1)
            self.apply_bcs(tn=dt, bc=bc, u=self.un)

        self.sol_data[1] = self.un.copy()

        for n in range(2, Nt+1):
            self.unp1[:] = 2*self.un - self.unm1 + C**2 * (D @ self.un)
            self.apply_bcs(tn=n*dt, bc=bc, u=self.unp1)
            self.unm1[:] = self.un
            self.un[:] = self.unp1

            self.sol_data[n] = self.unp1.copy()
    
    def plot_result(self,time):

        Nt = int(time/self.dt)
        t_n = np.linspace(0,time,Nt+1)
        xx,tt = np.meshgrid(problem.x,t_n,indexing='ij')

        analytic_sol = self.u_exact()

        self.fd_solution = fd_sol = np.zeros((self.N+1,Nt+1))
        analytic = np.zeros_like(fd_sol)
        for n in range(len(t_n)):

            self.fd_solution[:,n] = self.sol_data[n]
            analytic_sol_tn = sp.lambdify(x, analytic_sol.subs({L: self.LN, c: self.c, t: t_n[n]}))
            analytic[:,n] = analytic_sol_tn(self.x)

        fd_sol = self.fd_solution
        self.abs_diff = np.abs(fd_sol - analytic)

        plot2D(tt,xx,self.fd_solution,labels=['FD solution','t','x','u(x,t)'])
        plot2D(tt,xx,analytic,labels=['Analytic solution','t','x','u(x,t)'])
        plot2D(tt,xx,self.abs_diff,labels=['Difference','t','x','u(x,t)'])
        
    def u_exact(self, amp=1.):
        return amp*sp.sin(sp.pi*x/L)*sp.cos(c*sp.pi*t)

if __name__ == '__main__':
    test = 'wave1d'#,'diffusion1d'
    #test = 'diffusion1d'
    show = True
    print('PDE:',test)
    if test == 'wave1d':
        Nx = 10
        time = 1
        cfl = 0.01 
        domain = [0,1]  # L_0 and L_N
        c = 1.0
        A0 = 1.
        u0 = A0*sp.sin(sp.pi*x/L)
        bc = {'left': 0, 'right': 0}

        problem = Wave1DSolver(Nx,domain,cfl,u0,c)

        problem.solver(time,cfl=1,bc=bc,ic=0)

        problem.plot_result(time)

    elif test == 'diffusion1d':
        Nx = 10
        time = 1
        cfl = 0.01 
        domain = [0,1] 
        A0 = 1.
        u0 = A0*sp.sin(sp.pi*x/L)
        Df = 0.5
        bc = {'left': 0, 'right': 0}

        problem = Diffusion1DSolver(N=Nx,domain=domain,cfl=cfl,u0=u0,D=Df)

        problem.solver(time=time,cfl=cfl,bc=bc)

        problem.plot_result(time=time)

    if show:
        plt.show()



#class DiffusionFD