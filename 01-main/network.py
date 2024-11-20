from PDEq import *
from support import *

from autograd import jacobian, hessian, grad, elementwise_grad
import autograd.numpy as anp

from copy import copy

class FFNNetwork:

    def __init__(self,
                 layer_output_size,
                 activation_functions,
                 activation_derivatives,
                 PDE: Functions,
                 source_function,
                 domain_array,
                 domain=[0,1],
                 random_state = 1,
                 ):
        
        self.layer_out_size = layer_output_size
        self.act_func       = activation_functions
        self.act_der        = activation_derivatives
        self.PDE            = PDE
        self.source         = source_function
        self.domain         = domain
        self.domain_array   = domain_array
        self.random_seed    = random_state

        self.gd_method = list()

        anp.random.seed(self.random_seed)


    def create_layers(self): # Assemble P here, needs modification

        ## Number of layers, including the output layer
        num_layers = anp.size(self.layer_out_size)
        #print('num_layer',num_layers)

        ## Layer structure
        self.layers = [None]*(num_layers)

        ## Populating layers with uniform random numbers
        i_size = self.PDE.dimension + 1 # +1 to include the bias
        for l,layer in enumerate(self.layer_out_size):
            self.layers[l] = anp.random.randn(layer,i_size)
            i_size = layer + 1

        P = self.layers
        return P
    
    def feed_forward(self, P, point): # This is done for each time the trail function is evaluated, so this has to take in
                                      # the current version of P
        """ One feed-forward pass for one point in the input array. The method is called during 
            training to compute the value of the trail-function, used in the computation of the
            gradient for that step
        
        """

        point = point.reshape(anp.size(point,0),-1)
        num_points = anp.size(point,1)

        x_l = point; x_lm1 = x_l
        # loop over self.layers
        for W, act_func in zip( P, self.act_func ):
            
            ## Layer l-1
            x_lm1 = anp.concatenate( (anp.ones((1,num_points)), x_lm1), axis=0 )

            ## Activation layer l
            z_l = W @ x_lm1

            ## Output layer l
            x_l = act_func(z_l)

            ## Recasting layer l output for next loop
            x_lm1 = x_l
        
        return x_l[0][0]
    
    
    def trail_solution(self, point, P): # Specific trail function to 1d-diffusion problem
        self.PDE.trail_function(point,self.domain)
        return self.PDE.h1 + self.PDE.B * self.feed_forward(P, point)
    
    def cost_function(self, P): # Specific to u = u(x,t)
        """
        Computes the cost function that is minimized during training. Gradient of the cost function
        is done using ` autograd.grad() `. 

        Parameters
        ---
        P : list
            List of network layers

        Returns
        ---
        The computed cost for a given point

        """
        t,x = self.domain_array

        sum_cost = 0

        trail_jac_func = jacobian(self.trail_solution)
        trail_hess_func = hessian(self.trail_solution)

        #i = 0
        for tn in t:#self.domain_array[0]:
            for xi in x:#self.domain_array[-1]:
                point_in = anp.array([tn,xi])

                trail_jac  = trail_jac_func(point_in, P)
                trail_hess = trail_hess_func(point_in, P)

                trail_d = trail_jac[1]
                trail_d2 = trail_hess[0][0]

                f = self.PDE.right_hand_side(point_in,source_function=self.source)

                err_sqr = ( (trail_d - trail_d2) - f )**2

                sum_cost += err_sqr
        
        return sum_cost/ (anp.size(self.domain_array[0])*anp.size(self.domain_array[-1]))
    

    def train_network(self, GDMethod: GDTemplate,
                      learn_rate=0.01, epochs=100):

        ## Setup parameter matrix (Initial parameter matrix)
        P = self.create_layers()

        ## Assigning individual gradient descent objects to each layer
        for i in range(len(self.layers)):
            self.gd_method.append(copy(GDMethod))

        self.initial_cost = self.cost_function(P)
        print('Initial cost: %1.5e' %(self.initial_cost))

        ## Autograd-gradient function with the gradients of the layers in P
        autograd_cost_func = grad(self.cost_function,0)

        ## Training
        for e in range(epochs):
            grad_cost = autograd_cost_func(P)  # Current version of P, i.e. P(e)
            
            i = 0       # Counter for the gradient descent method
            for layer in range(len(P)):
                ## Implement with general gradient descent method
                P[layer] -= self.gd_method[i].update_change(grad_cost[layer],P[layer])
                #P[layer] -= learn_rate * grad_cost[layer]  # Learn-rate to be changed out to include scheduler

                i += 1

            if e % int(epochs/10) == 0:
                print('Epoch %i: Current cost: %1.5e' %(e,self.cost_function(P)))
            
            ## Resetting instances variables for the SGD-methods
            for gd in self.gd_method:
                gd.reset()


        self.final_cost = self.cost_function(P)
        print('Final cost: %1.5e' %(self.final_cost))
        self.fin_P_matrix = P

        return P 
    
    def evaluate(self): # Currently specific to u = u(x,t)
        t,x = self.domain_array
        Nx,Nt = anp.size(x), anp.size(t)

        ## Storing values from final solution parameter matrix and analytical solution
        self.network_solution = np.zeros((Nt,Nx))
        self.analytic = np.zeros_like(self.network_solution)
        for n, tn in enumerate(t):
            for i, xi in enumerate(x):
                point = np.array([tn,xi])

                self.network_solution[n,i] = self.trail_solution(point=point,P=self.fin_P_matrix)

                self.analytic[n,i] = self.PDE.analytical(domain_array=(point[1],point[0]),domain=self.domain)

        self.abs_diff = anp.abs(self.network_solution - self.analytic)
    
    def plot_result(self,plots='all',save=False,
                    f_name=['gen_name_1','gen_name_2','gen_name_3']):
        """ Plotting the results from the training, comparing the network solution
            to the analytical solution. 
            Gives 2D-surfaces of the development over time, contour-plots?, and 1D-plots
            at given time instances

            Parameters
            ---
            plots : str
                Keywords: 'all', 'network', 'exact', 'slices'
            save : bool
                If True, a figure with name *f_name* saves to the current directory
            f_name : list, str
                List of strings defining the names of the different 2D-plot figures
            
            Returns
            ---
            Nothing
        """
        self.evaluate()
        t,x = self.domain_array
        tt,xx = anp.meshgrid(t,x)

        ## Surface plots
        if plots == 'all' or plots == 'network':
            plot2D(xx,tt,self.network_solution,
                labels=['Network solution','t','x','u(x,t)'],
                save=save,f_name=f_name[1])
        
        if plots == 'all' or plots == 'exact':
            plot2D(xx,tt,self.analytic,
                labels=['Analytical solution','t','x','u(x,t)'],
                save=save,f_name=f_name[0])
            
        if plots == 'all' or plots == 'network':
            plot2D(xx,tt,self.abs_diff,
                labels=['Difference','t','x','u(x,t)'],
                save=save,f_name=f_name[2])
        
        ## Line plots showing different slices of surface defined by x,t
        if plots == 'slices':
            idx = [0,int(anp.size(t)/8),int(anp.size(t)/2),anp.size(t)-1]
            t_id = []; net_sol = []; analytic_res = []; fig, ax = [],[]
            fig,ax = plt.subplots(1,len(idx))
            fig.suptitle('Solutions at different times')
            for i in range(len(idx)):
                t_id.append(t[idx[i]])
                net_sol.append(self.network_solution[:,idx[i]])
                analytic_res.append(self.analytic[:,idx[i]])
                
                ax[i].plot(x,net_sol[i],label=r'$\tilde{u}$',lw=2.5)
                ax[i].plot(x,analytic_res[i],'--',label=r'$u_{e}$',)
                ax[i].set_title('t = %g' %t_id[i])
                ax[i].set_xlabel('x'); 
            
            ax[0].legend()
            ax[0].set_ylabel('u(x,t)',rotation=0,labelpad=15)
            fig.tight_layout()


## Class test case
if __name__ == "__main__":

    ## Random seed
    default_seed = 1; anp.random.seed(default_seed)

    ## Figure defaults
    plt.rcParams["figure.figsize"] = (8,3); plt.rcParams["font.size"] = 10

    anp.random.seed(default_seed)

    layer_out_sizes = [20,20,20,20,20,20,1]

    hidden_func = tanh # sigmoid ReLU, ELU, LeakyReLU,identity, tanh
    hidden_der  = elementwise_grad(hidden_func,0)

    act_funcs = []; act_ders = []
    for i in range(len(layer_out_sizes)-1):
        act_funcs.append(hidden_func)
        act_ders.append(hidden_der)
    act_funcs.append(identity); 
    output_der = identity #elementwise_grad(act_funcs[-1],0);
    act_ders.append(output_der)

    gd_method = ADAM(learning_rate=0.01,lmbda=0.01)

    #PDE = Diffusion1D(D=1.)
    PDE = Wave1D()
    f = PDE.right_hand_side

    x = anp.linspace(-1,1,10)
    t = anp.linspace(0,1,10)
    domain_array = anp.array([t,x])

    network = FFNNetwork(layer_out_sizes,act_funcs,act_ders,PDE,f,domain_array)
    #network.create_layers()
    P = network.train_network(GDMethod=gd_method,learn_rate=0.0001,epochs=50)
    network.evaluate()
    network.plot_result()
    plt.show()

    print(anp.max(network.abs_diff))