import autograd.numpy as anp
import tensorflow as tf
import numpy as np


class Functions:
    """Template class for the different PDE-classes and their specific functions"""

    def __init__(self,
                 sim_type='own',
                 rhs=0.
                 ):
        
        self.sim_type = sim_type
        self.rhs = rhs

        self.DTYPE = 'float32'
        if sim_type == 'flow':
            tf.keras.backend.set_floatx(self.DTYPE)

    def init_function(self,domain_array,domain):
        raise RuntimeError('Function not implemented in base class')
    
    def right_hand_side(self,domain_array,source_function):
        """The RHS forcing term of the PDE. Uses source_function to set value at point = (x1,x2,...,t)
           General for all PDE-classes"""
        return 0.
        #return source_function(point)

    def residual_function(self,u):
        raise RuntimeError('Function not implemented in base class')
    
    def analytical(self,domain_array,domain):
        raise RuntimeError('Function not implemented in base class')
    
class Burger1D(Functions):

    def __init__(self, sim_type='flow', rhs=0):
        super().__init__(sim_type, rhs)
        self.dimension = 2
        self.pi = tf.constant(np.pi,dtype=self.DTYPE)

    def init_function(self, domain_array, domain):
        x = domain_array[:,1:2]
        if self.sim_type == 'own':
            L = domain[1]
            return -anp.sin(anp.pi*x/L)
        elif self.sim_type == 'flow':
            print('flow')
            print('x',x.shape)
            L = tf.constant(domain[1],dtype=self.DTYPE)
            #return -tf.sin(self.pi * x/L)
            return -tf.sign(x)
    

    def boundary_function(self,domain_array,domain):
        t = domain_array[:,0]; x = domain_array[:,1]
        n = x.shape[0]
        if self.sim_type == 'own':
            return anp.zeros((n,1))
        #return self.analytical(domain_array=(x,t),domain=domain)
        elif self.sim_type == 'flow':
            return tf.zeros((n,1),dtype=self.DTYPE)

    
    def residual_function(self,u): #,u_t,u_x,u_xx):
        """ PDE's LHS. u = [u, u_t, u_x, u_tt, u_xx]
            Burger's equation: r = u_t + u*u_x - \\mu*u_xx"""
        mu = 0.01/self.pi
        return u[1] + u[0]*u[2] - mu * u[4]
    
    def analytical(self, domain_array, domain):
        x,t = domain_array; 
        if self.sim_type == 'own':
            L = domain[1]; 
            mu = 0.01/anp.pi
            eig = (anp.pi**2*mu)/(2.*L**2)
            return -2.*(anp.pi*mu*anp.cos(anp.pi*x/L))/(anp.sin(anp.pi*x/L) + anp.exp(eig*t))
            #return anp.sin(anp.pi*x/L)
        elif self.sim_type == 'flow':
            L = tf.constant(domain[1],dtype=self.DTYPE)
            mu = 0.01/self.pi
            #eig = (self.pi**2*mu)/(2.*L**2)
            return -tf.sin(self.pi*x)*tf.exp(-mu*self.pi**2*t)
            #return -2.*(self.pi*mu*tf.cos(self.pi*x/L))/(tf.sin(self.pi*x/L) + tf.exp(eig*t))
            #return -tf.sin(self.pi*x/L)

    
class Diffusion1D(Functions):
    """ One-dimensional diffusion equation, solving:
            du(x,t)/dt = du²(x,t)/dt
        using Dirichlet boundary conditions"""

    def __init__(self, sim_type='own', rhs=0, amp=1, D=1.):
        super().__init__(sim_type, rhs)

        self.sim_type = sim_type
        self.dimension = 2
        
        if sim_type == 'flow':
            self.Df = tf.constant(D, dtype=self.DTYPE)
            self.amp = tf.constant(amp, dtype=self.DTYPE)
            self.pi = tf.constant(np.pi,dtype=self.DTYPE)
        else:
            self.Df = D
            self.amp = amp

    def init_function(self,domain_array,domain):
        #x,t = domain_array[:,0], domain_array[:,1]
        x,t = domain_array[:,1:2], 0
        return self.analytical(domain_array=(x,t),domain=domain)
        #if self.sim_type == 'own':
        #    L = domain[1]
        #    return self.amp * anp.sin(anp.pi*x/L)
        #elif self.sim_type == 'flow':
        #    L = tf.constant(domain[1],dtype=self.DTYPE)
        #    return self.amp * tf.sin(self.pi * x/L)
    
    def boundary_function(self,domain_array,domain):
        x,t = domain_array[:,1], domain_array[:,0]
        x,t = domain_array[:,1:2], domain_array[:,0:1]
        return self.analytical((x,t),domain)
    

    def trail_function(self,point,domain):
        x,t = point; x0,xN = domain
        self.h1 = (1 - t) * (self.init_function(x) - ((1 - x/xN)*self.init_function(x0)) +
                                                      (x/xN)*self.init_function(xN))
        self.B = t * (1 - x/xN)*(x/xN)

    def residual_function(self,u): #, u_t, u_x, u_xx): # The PDE's LHS
        """ PDE's LHS. u = [u, u_t, u_x, u_tt, u_xx]
            Diffusion equation: r = u_t - D*u_xx"""
        return u[1] - self.Df*u[4]


    def analytical(self,domain_array,domain):
        x,t = domain_array#[:,0], domain_array[:,1]
        if self.sim_type == 'own':
            L = domain[1]; Df = self.Df
            return self.amp * anp.exp(-anp.pi**2 * Df * t/(L**2)) * anp.sin(anp.pi*x/L)
        elif self.sim_type == 'flow':
            L = tf.constant(domain[1],dtype=self.DTYPE); Df = self.Df
            return self.amp * tf.exp(-self.pi**2 * Df * t/(L**2)) * tf.sin(self.pi*x/L)
    
class Diffusion2D(Functions):

    def __init__(self, sim_type='own', rhs=0, amp=1., D=1.):
        super().__init__(sim_type, rhs)
        self.dimension = 3

        if self.sim_type == 'flow':
            self.Df = tf.constant(D, dtype=self.DTYPE)
            self.amp = tf.constant(amp, dtype=self.DTYPE)
            self.pi = tf.constant(np.pi,dtype=self.DTYPE)
        else:
            self.Df = D
            self.amp = amp

    def init_function(self,domain_array,domain):
        #t,x,y = domain_array[:,0:1], domain_array[:,1:2], domain_array[:,2:3]
        t,x,y = 0, domain_array[:,1:2], domain_array[:,2:3]
        '''Lx,Ly = domain[0][1],domain[1][1]
        if self.sim_type == 'flow':
            Lx,Ly = tf.constant(Lx,dtype=self.DTYPE), tf.constant(Ly, dtype=self.DTYPE)
            return self.amp * tf.sin(self.pi*x/Lx)*tf.sin(self.pi*y/Ly)'''
        return self.analytical(domain_array=(x,y,t),domain=domain) 
    
    
    def trail_function(self,domain_array,domain):
        raise RuntimeError
    
    def boundary_function(self,domain_array,domain):
        t,x,y = domain_array[:,0:1], domain_array[:,1:2], domain_array[:,2:3]
        return self.analytical(domain_array=(x,y,t),domain=domain)
    
    def residual_function(self, u):
        """ PDE's LHS. u = [u, u_t, u_x, u_y, u_tt, u_xx, u_yy]
            Diffusion equation: r = u_t - D*(u_xx + u_yy"""
        return u[1] - self.Df*(u[5] - u[-1])
    
    def analytical(self,domain_array,domain):
        x,y,t = domain_array #domain_array[:,0], domain_array[:,1], domain_array[:,2]
        Lx,Ly = domain[0][1],domain[1][1]
        if self.sim_type == 'own':
            eig = self.Df*anp.pi**2*(1/Lx**2 + 1/Ly**2)
            return self.amp * anp.sin(anp.pi*x/Lx)*anp.sin(anp.pi*y/Ly)*anp.exp(-eig*t)
        elif self.sim_type == 'flow':
            Lx,Ly = tf.constant(Lx,dtype=self.DTYPE), tf.constant(Ly, dtype=self.DTYPE)
            eig = tf.constant(self.Df*self.pi**2*(1/Lx**2 + 1/Ly**2), dtype=self.DTYPE)
            return self.amp * tf.sin(self.pi*x/Lx)*tf.sin(self.pi*y/Ly)*tf.exp(-eig*t)
    
class Wave1D(Functions):

    def __init__(self, sim_type='own', rhs=0,amp=1, c=1):
        super().__init__(sim_type, rhs)

        self.dimension = 2
        
        if self.sim_type == 'flow':
            self.c = tf.constant(c, dtype=self.DTYPE)
            self.amp = tf.constant(amp, dtype=self.DTYPE)
            self.pi = tf.constant(np.pi,dtype=self.DTYPE)
        else:
            self.c = c
            self.amp = amp

    def init_function(self,domain_array,domain): #u = u(x,t = 0)
        x,t = domain_array[:,1:2], 0
        if self.sim_type == 'own':
            L = domain[1]
            return self.amp * anp.sin(anp.pi*x/L)
        elif self.sim_type == 'flow':
            L = tf.constant(domain[1],dtype=self.DTYPE)
            return self.amp * tf.sin(self.pi*x/L)
        
    def boundary_function(self,domain_array,domain):
        x,t = domain_array[:,1:2], domain_array[:,0:1]
        return self.analytical((x,t),domain)
    
    def trail_function(self,point,domain):
        x,t = point; x0,xN = domain
        u0_dt = -self.amp*self.c*anp.pi*anp.sin(anp.pi*x/xN)*anp.sin(self.c*anp.pi*t)
        self.h1 = (1 - t**2) * (self.init_function(x) -
                ((1 - x/xN)*self.init_function(x0)) + (x/xN)*self.init_function(xN)) + t*u0_dt
        self.B = t**2 * (1 - (x/xN)) * (x/xN)

    def residual_function(self,u):
        """ PDE's LHS. u = [u, u_t, u_x, u_tt, u_xx]
            Wave equation: r = u_tt - c²u_xx"""
        return u[3] - self.c**2 * u[4]
    
    def analytical(self,domain_array,domain):
        x,t = domain_array; 
        if self.sim_type == 'own':
            L = domain[1]
            return self.amp * anp.sin(anp.pi*x/L)*anp.cos(self.c*anp.pi*t)
        elif self.sim_type == 'flow':
            L = tf.constant(domain[1],dtype=self.DTYPE)
            return self.amp * tf.sin(self.pi*x/L)*tf.cos(self.c*self.pi*t)
    
    
class Wave2D(Functions):

    def __init__(self, sim_type='own', rhs=0,amp=1, c=1, m=[1,1]):
        super().__init__(sim_type, rhs)

        self.dimension = 2
        
        if self.sim_type == 'flow':
            self.c = tf.constant(c, dtype=self.DTYPE)
            self.amp = tf.constant(amp, dtype=self.DTYPE)
            self.pi = tf.constant(np.pi,dtype=self.DTYPE)
            self.mx = tf.constant(m[0],dtype=self.DTYPE)
            self.my = tf.constant(m[1],dtype=self.DTYPE)
        else:
            self.c = c
            self.amp = amp
            self.mx = m[0]; self.my = m[1]

    def init_function(self,domain_array,domain): #u = u(x,t = 0)
        t,x,y = domain_array[:,0:1], domain_array[:,1:2], domain_array[:,2:3]
        return self.analytical((x,y,t),domain)
        '''if self.sim_type == 'own':
            L = np.abs(domain[1] - domain[0])
            return self.amp * anp.sin(anp.pi*x/L)
        elif self.sim_type == 'flow':
            L = tf.constant(np.abs(domain[1] - domain[0]),dtype=self.DTYPE)
            return self.amp * tf.sin(self.pi*x/L)'''
        
    def boundary_function(self,domain_array,domain):
        t,x,y = domain_array[:,0:1], domain_array[:,1:2], domain_array[:,2:3]
        return self.analytical((x,y,t),domain)
    
    def trail_function(self,point,domain):
        raise NotImplementedError
        '''x,t = point; x0,xN = domain
        u0_dt = -self.amp*self.c*anp.pi*anp.sin(anp.pi*x/xN)*anp.sin(self.c*anp.pi*t)
        self.h1 = (1 - t**2) * (self.init_function(x) -
                ((1 - x/xN)*self.init_function(x0)) + (x/xN)*self.init_function(xN)) + t*u0_dt
        self.B = t**2 * (1 - (x/xN)) * (x/xN)'''

    def residual_function(self,u):
        """ PDE's LHS. u = [u, du_dt, du_dx, du_dy, du_dtt, du_dxx, du_dyy]
            Wave equation: r = du_dtt - c²/du_dxx + du_dyy)"""
        return u[4] - self.c**2 * (u[5] - u[-1])
    
    def analytical(self,domain_array,domain):
        x,y,t = domain_array; 
        if self.sim_type == 'own':
            L = anp.abs(domain[1] - domain[0])
            kx,ky = self.mx*anp.pi, self.my*anp.pi
            k = anp.max(kx,ky); w = k*self.c
            return self.amp * anp.sin(kx*x/L)*anp.sin(ky*y/L)*anp.cos(w*t)
        elif self.sim_type == 'flow':
            Lx = tf.constant(np.abs(domain[0][1] - domain[0][0]), dtype=self.DTYPE)
            Ly = tf.constant(np.abs(domain[1][1] - domain[1][0]), dtype=self.DTYPE)
            print(Lx)
            kx,ky = tf.constant(self.mx*self.pi,dtype=self.DTYPE), tf.constant(self.my*self.pi,dtype=self.DTYPE)
            k = max(kx,ky)
            w = tf.constant(k*self.c, dtype=self.DTYPE)
            return self.amp * tf.sin(kx*x/Lx)*tf.sin(ky*y/Ly)*tf.cos(w*t)