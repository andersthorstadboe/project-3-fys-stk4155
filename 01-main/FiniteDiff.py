from support import *

import numpy as np
from scipy import sparse
import sympy as sp
import matplotlib.pyplot as plt

x,y,t,c,L = sp.symbols('x,t,y,c,L')

class FDSolver:
    """Template class for the finite difference solvers"""

    def __init__(self,
                 N,
                 domain=[0.,1.],
                 cfl=1,
                 u0=None):
        
        self.N   = N
        self.L0  = domain[0]
        self.LN  = domain[1]
        self.L   = self.LN + np.abs(self.L0)
        self.cfl = cfl
        self.u0  = u0
        self.dx  = domain[1]/N
        self.x   = np.linspace(domain[0],domain[1],N+1)

    @property
    def dt(self):
        raise RuntimeError

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
    
    def apply_bcs(self, bc, u=None):
        raise RuntimeError
    
    def solver(self,Nt,cfl=None, bc=0, ic=0):
        raise RuntimeError
    
    def evaluate(self,tN=1):
        raise RuntimeError
    
    def plot_result(self,tN=1,
                    plots='all',
                    idx=[0,-1],
                    save=False,
                    f_name=['gen_name_1','gen_name_2','gen_name_3']
                    ):

        self.evaluate(tN=tN)

        xx,tt = np.meshgrid(self.x,self.t_n, indexing='ij')

        if plots == 'all' or plots == 'finite':
            plot2D(tt,xx,self.fd_solution,
                labels=['FD solution','t','x','u(x,t)'],
                save=save,f_name=f_name[0])
        if plots == 'all' or plots == 'exact':
            plot2D(tt,xx,self.analytic,
                labels=['Analytic solution','t','x','u(x,t)'],
                save=save,f_name=f_name[1])
        if plots == 'all' or plots == 'finite':
            plot2D(tt,xx,self.abs_diff,
                labels=['Difference','t','x','u(x,t)'],
                save=save,f_name=f_name[2])
            
        ## Line plots showing different slices of surface defined by x,t
        if plots == 'slices' or plots == 'all':
            t_id = []; fd_sol = []; analytic_res = []; 
            fig,ax = plt.subplots(1,len(idx))
            fig.suptitle('Solutions at different times')
            for i in range(len(idx)):
                t_id.append(self.t_n[idx[i]])
                fd_sol.append(self.fd_solution[:,idx[i]])
                analytic_res.append(self.analytic[:,idx[i]])
                
                ax[i].plot(self.x,fd_sol[i],label=r'$\tilde{u}$',lw=2.5)
                ax[i].plot(self.x,analytic_res[i],'--',label=r'$u_{e}$',)
                ax[i].set_title('t = %g' %t_id[i])
                ax[i].set_xlabel('x'); 
            
            ax[0].legend()
            ax[0].set_ylabel('u(x,t)',rotation=0,labelpad=15)
            fig.tight_layout()
    
    def u_exact(self):
        raise RuntimeError
    

class Diffusion1DSolver(FDSolver):

    def __init__(self, N, domain, cfl, u0, amp=1., D=1.):
        super().__init__(N, domain, cfl, u0)
        self.Df = D
        self.amp = amp

        self.unp1 = np.zeros(N+1)
        self.un = np.zeros(N+1)

    @property
    def dt(self):
        return self.cfl*self.dx**2/self.Df 


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
            u[0] = self.ue.subs({L: self.L, x: self.L0, t: tn})
        elif bcl == 1:
            u[0] = self.un[0] + self.cfl*(2*self.un[1]-2*self.un[0]) - ...
            ... - (self.cfl/self.dx)*ue_dx.subs({L: self.L, x: self.L0, t: tn}) + ... 
            + self.dt*self.ue.subs({L: self.L, x: self.L0, t: tn})

        ## Right boundary
        if bcr == 0:
            u[-1] = self.ue.subs({L: self.L, x: self.LN, t: tn})
        elif bcr == 1:
            u[-1] = self.un[-1] + self.cfl*(2*self.un[-2]-2*self.un[-2]) - ...
            ... - (self.cfl/self.dx)*ue_dx.subs({L: self.L, x: self.LN, t: tn}) + ...
            ... + self.dt*self.ue.subs({L: self.L, x: self.LN, t: tn})

        return u
    
    def solver(self, tN=1, cfl=None, bc={'left': 0, 'right': 0}, ic=0):

        self.ue = self.u_exact()

        ## Initialization of D²-matrix
        D = self.D2(bc)

        ## Setting Courant number, time-step and initial solution
        #self.cfl = C = self.cfl if cfl is None else self.c*self.dt/self.dx
        self.cfl = C = self.Df*self.dt/self.dx**2
        dt = self.dt
        Nt = int(tN/dt)

        ## Setting initical condition for u_n-array
        u0 = sp.lambdify(x, self.u0.subs({L: self.L, t: 0}))
        self.un[:] = u0(self.x)

        self.sol_data = {0: self.un.copy()}

        for n in range(1,Nt+1):
            self.unp1[:] = self.un + C* (D @ self.un)
            self.unp1 = self.apply_bcs(tn=n*self.dt, bc=bc, u=self.unp1)
            
            ## Assigning updated solution for next timestep
            self.un[:] = self.unp1

            self.sol_data[n] = self.unp1.copy()

    def evaluate(self, tN=1):

        Nt = int(tN/self.dt)
        self.t_n = np.linspace(0,tN,Nt+1)

        self.fd_solution = np.zeros((self.N+1,Nt+1))
        self.analytic = np.zeros_like(self.fd_solution)
        for n in range(len(self.t_n)):

            self.fd_solution[:,n] = self.sol_data[n]
            analytic_sol_tn = sp.lambdify(x, self.ue.subs({L: self.L, t: self.t_n[n]}))
            self.analytic[:,n] = analytic_sol_tn(self.x)
        
        self.abs_diff = np.abs(self.fd_solution - self.analytic)


    def u_exact(self):
        """ Analytic solution implemented as a Sympy-function """
        amp = self.amp
        Df = self.Df
        return amp * sp.exp(-sp.pi**2 * Df * t/(L**2)) * sp.sin(sp.pi * x/L)
    

class Wave1DSolver(FDSolver):

    def __init__(self, N, domain, cfl, u0,amp=1.,c=1.):
        super().__init__(N, domain, cfl, u0)
        self.c = c
        self.amp = amp

        self.unp1 = np.zeros(N+1)
        self.un = np.zeros(N+1)
        self.unm1 = np.zeros(N+1)

    @property
    def dt(self):
        return self.cfl*self.dx/self.c


    def apply_bcs(self, t_n, bc, u=None):
        u = u if u is not None else self.unp1
        bcl, bcr = bc['left'], bc['right']
        ue_dx = sp.diff(self.ue, x, 1)
        
        ## Boundary conditions set for arb. RHS, f(x,t) for x = {0, L}
        ## Left boundary
        if bcl == 0:    # Dirichlet condition, left, set in D²-matrix
            u[0] = self.ue.subs({L: self.L, c: self.c, x: self.L0, t: t_n})
        elif bcl == 1:  # Neumann condition, left, set in D²-matrix
            u[0] = 2*self.un[0] - self.unm1[0] + self.cfl**2*(2*self.un[1] - self.un[0]) - ...
            ... - 2*self.cfl**2*self.dx*ue_dx.subs({L: self.L, c: self.c, x: self.L0, t: t_n}) + ...
            ... + self.dt**2*self.ue.subs({L: self.L, c: self.c, x: self.L0, t: t_n})

        # Right boundary
        if bcr == 0: # Dirichlet condition, right, set in D²-matrix
            u[-1] = self.ue.subs({L: self.L, c: self.c, x: self.LN, t: t_n})
        elif bcr == 1: # Neumann condition, right, set in D²-matrix
            u[-1] = 2*self.un[-1] - self.unm1[-1] + self.cfl**2*(2*self.un[-2] - self.un[-1]) + ...
            ... + 2*self.cfl**2*self.dx*ue_dx.subs({L: self.L, c: self.c, x: self.LN, t: t_n}) + ...
            ... + self.dt**2*self.ue.subs({L: self.L, c: self.c, x: self.LN, t: t_n})

    def solver(self, tN=1., cfl=None, bc=0, ic=0):

        self.ue = self.u_exact()

        ## Initialization of D²-matrix
        D = self.D2(bc)

        ## Setting Courant number, time-step and initial solution
        #self.cfl = C = self.cfl if cfl is None else self.c*self.dt/self.dx
        self.cfl = C = self.c*self.dt/self.dx
        dt = self.dt
        Nt = int(tN/dt)

        ## First step. Set u_nm1-array
        u0 = sp.lambdify(x, self.u0.subs({L: self.L, c: self.c, t: 0}))
        self.unm1[:] = u0(self.x)

        ## Time-series storage
        self.sol_data = {0: self.unm1.copy()}

        ## Setting initial conditions for u_n-array,
        if ic == 0: # use sympy function
            u0 = sp.lambdify(x, self.u0.subs({L: self.L, c: self.c, t: dt}))
            self.un[:] = u0(self.x)
        else: 
            self.un[:] = self.unm1 + 0.5*C**2* (D @ self.unm1)
            self.apply_bcs(t_n=dt, bc=bc, u=self.un)

        self.sol_data[1] = self.un.copy()

        for n in range(2, Nt+1):
            self.unp1[:] = 2*self.un - self.unm1 + C**2 * (D @ self.un)
            self.apply_bcs(t_n=n*dt, bc=bc, u=self.unp1)
            self.unm1[:] = self.un
            self.un[:] = self.unp1

            self.sol_data[n] = self.unp1.copy()

    def evaluate(self, tN=1):

        Nt = int(tN/self.dt)
        self.t_n = np.linspace(0,tN,Nt+1)

        self.fd_solution = fd_sol = np.zeros((self.N+1,Nt+1))
        self.analytic = np.zeros_like(fd_sol)
        for n in range(len(self.t_n)):

            self.fd_solution[:,n] = self.sol_data[n]
            analytic_sol_tn = sp.lambdify(x, self.ue.subs({L: self.L, c: self.c, t: self.t_n[n]}))
            self.analytic[:,n] = analytic_sol_tn(self.x)
        
        self.abs_diff = np.abs(fd_sol - self.analytic)  
        
    def u_exact(self):
        amp = self.amp
        return amp*sp.sin(sp.pi*x/L)*sp.cos(c*sp.pi*t)
    

class Wave2DSolver(Wave1DSolver):

    def __init__(self, N, domain=([0, 1],[0, 1]),
                 cfl=1, amp=1, c=1,
                 m = [1,1]):

        self.N = N
        self.Lx = domain[0][1] + np.abs(domain[0][0])
        self.Ly = domain[1][1] + np.abs(domain[1][0])
        self.x0,self.xN = domain[0][0],domain[0][1]
        self.y0,self.yN = domain[1][0],domain[1][1]
        self.mx = m[0]; self.my = m[1]
        self.cfl = cfl
        self.c = c
        self.amp = amp

        self.x,self.y = np.linspace(self.x0,self.xN,N+1),np.linspace(self.y0,self.yN,N+1) 

    @property
    def dt(self):
        self.h_x, self.h_y = self.Lx/self.N, self.Ly/self.N
        h = max(self.h_x,self.h_y)
        return (self.cfl*h)/self.c
    
    @property
    def w(self):
        kx, ky = self.mx*np.pi, self.my*np.pi
        return self.c*np.sqrt(kx**2 + ky**2)
    
    def create_mesh(self, sparse=False):
        xi,yj = np.linspace(self.x0,self.xN,self.N+1),np.linspace(self.y0,self.yN,self.N+1)
        self.xij, self.yij = np.meshgrid(xi,yj,indexing='ij',sparse=sparse)


    def initialize(self,):
        self.Unp1,self.Un,self.Unm1 = np.zeros((3,self.N+1,self.N+1))
        self.Unm1[:] = sp.lambdify((x,y,t), self.ue)(self.xij,self.yij,0)
        self.Un[:] = self.Unm1 + .5*(self.c*self.dt)**2*(self.D2x @ self.Unm1 + self.Unm1 @ self.D2y.T)


    def D2(self,bc=0):
        """Return second order differentiation matrix, unscaled"""
        bc1, bc2 = 0, 0 #bc['0'], bc['N']
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        if bc1 == 1:
            D[0, :4] = -2, 2, 0, 0
        if bc2 == 1:
            D[-1, -4:] = 0, 0, 2, -2

        return D

    def apply_bcs(self, U, bc):
        """ Applying Dirichlet boundary conditions with arbitrary RHS """
        bcl, bcr = bc['x0'], bc['xN']
        bcd, bcu = bc['y0'], bc['yN']
        #ue_dx, ue_dy = sp.diff(self.ue, x, 1), sp.diff(self.ue, y, 1)
        # x = 0
        if bcl == 0:
            U[0] = self.ue.subs({x: self.x0})
        #elif bcl == 1:
        #    U[0] = 0
        # x = Lx
        if bcr == 0:
            U[-1] = self.ue.subs({x: self.xN})
        #elif bcr == 1:
        #    U[-1] = 0
        # y = 0
        if bcd == 0:
            U[:,0] = self.ue.subs({y: self.y0})
        #elif bcd == 1:
        #    U[:,0] = 0
        # y = Ly
        if bcu == 0:
            U[:,-1] = self.ue.subs({y: self.yN})
        #elif bcu == 1:
        #    U[:,-1] = 0

        return U
    
    def solver(self,tN=1.,cfl=None, bc=0):
        
        ## Analytical solution for boundary and initial.
        self.ue = self.u_exact()
        
        ## Mesh and initialization
        self.cfl = cfl; self.c = c
        self.Nt = int(tN/self.dt)
        self.create_mesh(sparse=True)

        bc_keys = list(bc)
        bc_keys_x = bc_keys[0:2]; bc_keys_y = bc_keys[2:] 

        self.D2x = self.D2(bc=0)/self.h_x
        self.D2y = self.D2(bc=0)/self.h_y

        self.initialize()
        self.apply_bcs(self.Un,bc=bc)

        self.sol_data = {0: self.Unm1.copy(), 1: self.Un.copy()}

        for n in range(2,self.Nt+1):
            self.Unp1[:] = 2*self.Un - self.Unm1 + (self.c*self.dt)**2 * (self.D2x @ self.Un 
                                                                          + self.Un @ self.D2y.T)
            #Boundary condictions
            self.apply_bcs(self.Unp1,bc=bc)  
            # Updating Un, Unm1
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
        

            self.sol_data[n] = self.Unm1.copy()

    
    def evaluate(self, tN=1):
        
        self.t_n = np.linspace(0,tN,self.Nt+1)

        self.analytic = {}
        self.fd_solution = np.zeros((self.N+1,self.Nt+1))
        diff = []
        for n in range(len(self.t_n)):

            analytic_sol_tn = sp.lambdify((x,y), self.ue.subs({t: self.t_n[n]}))

            self.analytic[n] = analytic_sol_tn(self.xij,self.yij)

            diff.append(self.sol_data[n] - self.analytic[n])
            

        self.abs_diff = np.abs(diff)

    def plot_result(self,
                    tN=1,
                    plots='all',
                    idx=[0, -1],
                    save=False, 
                    f_name=['gen_name_1', 'gen_name_2', 'gen_name_3']):
        
        self.evaluate(tN=tN)

        xx,yy = self.xij,self.yij
        
        for id in idx:
            if plots == 'all' or plots == 'finite':
                plot2D(xx,yy,self.sol_data[id],
                    labels=['FD solution','x','y','u(x,y,t)'],
                    save=save,f_name=f_name[0])
            if plots == 'all' or plots == 'exact':
                plot2D(xx,yy,self.analytic[id],
                    labels=['Analytic solution','x','y','u(x,y,t)'],
                    save=save,f_name=f_name[1])
            if plots == 'all' or plots == 'finite':
                plot2D(xx,yy,self.abs_diff[id],
                    labels=['Difference','t','x','u(x,t)'],
                    save=save,f_name=f_name[2])


    def u_exact(self):
        return self.amp * sp.sin(self.mx*sp.pi*x)*sp.sin(self.my*sp.pi*y)*sp.cos(self.w*t)


class Diffusion2DSolver(Wave2DSolver):

    def __init__(self, N, domain=([0, 1], [0, 1]), cfl=1, amp=1, c=1, m=[1, 1]):
        super().__init__(N, domain, cfl, amp, c, m)

        self.Nx = N[0]; self.Ny = N[1]

        self.Lx = domain[0][1] + np.abs(domain[0][0])
        self.Ly = domain[1][1] + np.abs(domain[1][0])


    @property
    def dt(self):
        raise NotImplementedError
    
    def create_mesh(self, sparse=False):
        self.h_x, self.h_y = self.Lx/self.Nx, self.Ly/self.Ny
        xi,yj = np.linspace(0,self.Lx,self.Nx+1),np.linspace(0,self.Ly,self.Ny+1)
        self.xij, self.yij = np.meshgrid(xi,yj,indexing='ij',sparse=sparse) 
        #return self.xij, self.yij

    def D2(self,N):
        """Return second order differentiation matrix, unscaled"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        #if self.Lx == self.Ly:
        #    D /= self.h**2
        return D

    def apply_bcs(self, bc, u=None):
        raise NotImplementedError
    
    def solver(self,tN=1.,cfl=None, bc=0):
        raise NotImplementedError
    
    def evaluate(self, tN=1):
        raise NotImplementedError
    
    def plot_result(self, tN=1,
                    plots='all',
                    idx=[0, -1],
                    save=False,
                    f_name=['gen_name_1', 'gen_name_2', 'gen_name_3']):
        
        raise NotImplementedError
    
    def u_exact(self):
        raise NotImplementedError


if __name__ == '__main__':
    test = 'wave1d'
    #test = 'diffusion1d'
    test = 'wave2d'
    show = True
    print('PDE:',test)
    if test == 'wave1d':
        Nx = 10
        tN = 1
        cfl = 0.1 
        domain = [-1,1]  # L_0 and L_N
        c = 1.0
        A0 = 1.
        u0 = A0*sp.sin(sp.pi*x/L)
        bc = {'left': 0, 'right': 0}

        problem = Wave1DSolver(Nx,domain,cfl,u0,c)

        problem.solver(tN=tN,cfl=1,bc=bc,ic=0)

        problem.plot_result(tN=tN)
        print(problem.abs_diff.shape)

    elif test == 'diffusion1d':
        print('here')
        Nx = 10
        tN = 1
        cfl = 0.01 
        domain = [0,1] 
        A0 = 1.
        u0 = A0*sp.sin(sp.pi*x/L)
        Df = 0.5
        bc = {'left': 0, 'right': 0}

        problem = Diffusion1DSolver(N=Nx,domain=domain,cfl=cfl,u0=u0,amp=A0,D=Df)

        problem.solver(tN=tN,cfl=cfl,bc=bc)

        problem.plot_result(tN=tN)

    elif test == 'wave2d':
        N = 100
        tN = 1.
        cfl = 0.01
        domain = ([-1,1],[-1,1])
        c = 1.0
        amp = 1.
        m = [3,3]
        bc = {'x0': 0, 'xN': 0, 'y0': 0, 'yN': 0}

        problem = Wave2DSolver(N=N,domain=domain,
                               cfl=cfl, amp=amp, c=c, m=m)
        
        problem.solver(tN=tN,cfl=cfl,bc=bc)

        keys = list(problem.sol_data)

        print(len(keys))
        problem.evaluate(tN=1)
        print(problem.analytic[0].shape)
        print(np.max(problem.abs_diff))


        problem.plot_result(tN=tN,idx=[5,20])
        

    if show:
        plt.show()



#class DiffusionFD