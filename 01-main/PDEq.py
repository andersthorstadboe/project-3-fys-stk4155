import autograd.numpy as anp


class Functions:
    """Template class for the different PDE-classes and their specific functions"""

    def __init__(self,
                 point,
                 ):
        self.point = point

    def init_function(self,position):
        raise RuntimeError
    
    def right_hand_side(self,point,source_function):
        """The RHS forcing term of the PDE. Uses source_function to set value at point = (x1,x2,...,t)
           General for all PDE-classes"""
        return 0.
        #return source_function(point)
    
    def analytical(self,position):
        raise RuntimeError
    
class Diffusion1D(Functions):
    """ One-dimensional diffusion equation, solving:
            \partial_{t} u(x,t) = \partial_{xx}u(x,t)
        using Dirichlet boundary conditions
    """

    def __init__(self, point):
        super().__init__(point)
        self.dimension = 2


    def init_function(self,position):
        return anp.sin(position)

    def analytical(self,position):
        #print(position)
        return anp.exp(-anp.pi**2 * position[-1]) * anp.sin(anp.pi*position[0])
    
class Diffusion2D(Functions):

    def __init__(self, point):
        super().__init__(point)
        self.dimension = 3

    def init_function(self):
        raise RuntimeError
    
    
    def right_hand_side(self):
        raise RuntimeError
    
    def analytical(self):
        raise RuntimeError
    
class Wave1D(Functions):

    def __init__(self, point):
        super().__init__(point)
        self.dimension = 2


    def init_function(self):
        raise RuntimeError
    
    
    def right_hand_side(self):
        raise RuntimeError
    
    
    def analytical(self):
        raise RuntimeError
    
    
class Wave2D(Functions):

    def __init__(self, point):
        super().__init__(point)


    def init_function(self):
        raise RuntimeError
    
    
    def right_hand_side(self):
        raise RuntimeError
    
    
    def analytical(self):
        raise RuntimeError