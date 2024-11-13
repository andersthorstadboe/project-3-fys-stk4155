from PDEq import *
from support import *

from autograd import jacobian, hessian, grad, elementwise_grad
import autograd.numpy as anp

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


    def create_layers(self): # Assemble P here, needs modification
        #print('layers')
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

        point = point.reshape(anp.size(point,0),-1)
        num_points = anp.size(point,1)

        x_l = point; x_lm1 = x_l
        # loop over self.layers
        for W, act_func in zip( P, self.act_func ):
            x_lm1 = anp.concatenate( (anp.ones((1,num_points)), x_lm1), axis=0 )
            #print('W',W.shape)
            #print('x_l-1',x_lm1.shape)
            z_l = W @ x_lm1

            x_l = act_func(z_l)

            x_lm1 = x_l
        
        #print('x_l:',x_l.shape)
        #print('x_l',x_l[0][0])
        #print('ff_fin')
        return x_l[0][0]
    
    def back_propagation(self):
        raise NotImplementedError
    
    
    def trail_solution(self, point, P): # Specific trail function to diffusion probelm

        x,t = point; x0,xN = self.domain

        h1 = (1 - t) * (self.PDE.init_function(x) - ((1 - x/xN)*self.PDE.init_function(x0)) +
                                                      (x/xN)*self.PDE.init_function(xN))
        B  = t * (1 - x/xN)*(x/xN)
        return h1 + B * self.feed_forward(P, point)
    
    def cost_function(self, P): # Specific to u = u(x,t)
        """
        Computes the cost function that is minimized for the gradinet descent

        Parameters
        ---
        P : list
            ...
        domain_array : NDArray
            Array containing x_i in the i first columns, t in the last column

        Returns
        ---
        The computed cost 

        """

        sum_cost = 0

        trail_jac_func = jacobian(self.trail_solution)
        trail_hess_func = hessian(self.trail_solution)

        #i = 0
        for xi in self.domain_array[0]:
            for tn in self.domain_array[-1]:
                point_in = anp.array([xi,tn])

                trail_jac  = trail_jac_func(point_in, P)
                trail_hess = trail_hess_func(point_in, P)

                trail_d = trail_jac[1]
                trail_d2 = trail_hess[0][0]

                f = self.PDE.right_hand_side(point_in,source_function=self.source)

                err_sqr = ( (trail_d - trail_d2) - f )**2

                sum_cost += err_sqr
        
        return sum_cost/ (anp.size(self.domain_array[0])*anp.size(self.domain_array[-1]))
    

    def train_network(self,learn_rate=0.01, epochs=100):

        # Setup parameter matrix (Initial parameter matrix)
        P = self.create_layers()

        self.initial_cost = self.cost_function(P)
        print('Initial cost: %g' %(self.cost_function(P)))

        ## Autograd-gradient function with the gradients of the layers in P
        autograd_cost_func = grad(self.cost_function,0)
        #i = 0
        ## Training
        for e in range(epochs):
            grad_cost = autograd_cost_func(P)  # Current version of P, i.e. P(e)

            for layer in range(len(P)):
                P[layer] -= learn_rate * grad_cost[layer]  # Learn-rate to be changed out to include scheduler

            #print('i =',i)
            #i+=1

        self.final_cost = self.cost_function(P)
        print('Final cost:',self.final_cost)
        self.fin_P_matrix = P

        return P 
    
    def evaluate(self): # Currently specific to u = u(x,t)
        x,t = self.domain_array
        Nx,Nt = anp.size(x), anp.size(t)

        self.network_solution = np.zeros((Nx,Nt)); self.analytic = np.zeros_like(self.network_solution)

        for i, xi in enumerate(x):
            for n, tn in enumerate(t):
                point = np.array([xi,tn])

                self.network_solution[i,n] = self.trail_solution(point=point,P=self.fin_P_matrix)

                self.analytic[i,n] = self.PDE.analytical(position=point)

        self.abs_diff = anp.abs(self.network_solution - self.analytic)

        return 0
    
    def plot_result(self,save=False,f_name='gen_name.png'):
        """ Plotting the results from the training, comparing the network solution
            to the analytical solution. 
            Gives 2D-surfaces of the development over time, contour-plots?, and 1D-plots
            at given time instances
        """
        self.evaluate()
        x,t = self.domain_array
        xx,tt = anp.meshgrid(x,t)

        ## Surface plots
        fig0,ax0 = plot2D(tt,xx,self.analytic,
                          labels=['Analytical solution','t','x','u(x,t)'],
                          save=save,f_name=f_name)
        fig1,ax1 = plot2D(tt,xx,self.network_solution,
                          labels=['Network solution','t','x','u(x,t)'],
                          save=save,f_name=f_name)
        fig2,ax2 = plot2D(tt,xx,self.abs_diff,
                          labels=['Difference','t','x','u(x,t)'],
                          save=save,f_name=f_name)
        
        ## Line plots showing different slices of surface defined by x,t
        idx = [0,int(anp.size(t)/2),anp.size(t)-1]
        t_id = []; res = []; analytic_res = []; fig, ax = [],[]
        fig,ax = plt.subplots(1,len(idx))
        fig.suptitle('Solutions at different times')
        for i in range(len(idx)):
            t_id.append(t[idx[i]])
            res.append(self.network_solution[:,idx[i]])
            analytic_res.append(self.analytic[:,idx[i]])
            
            ax[i].plot(x,res[i],label=r'$\tilde{u}$',lw=2.5)
            ax[i].plot(x,analytic_res[i],'--',label=r'$u_{e}$',)
            ax[i].set_title('Solution at t = %g' %t_id[i])
            ax[i].set_xlabel('x'); ax[i].set_ylabel('u(x,%i)' %(int(t_id[i])),
                                                    rotation=0,labelpad=15)
            ax[i].legend()





    

if __name__ == "__main__":

    ## Random seed
    default_seed = 1; anp.random.seed(default_seed)

    ## Figure defaults
    plt.rcParams["figure.figsize"] = (8,3); plt.rcParams["font.size"] = 10

    anp.random.seed(default_seed)

    problem_dimension = 2

    net_in_size = 10
    layer_out_sizes = [2,1]

    hidden_func = sigmoid #ReLU #sigmoid ReLU, ELU, LeakyReLU,identity
    hidden_der  = elementwise_grad(hidden_func,0)

    act_funcs = []; act_ders = []
    for i in range(len(layer_out_sizes)-1):
        act_funcs.append(hidden_func)
        act_ders.append(hidden_der)
    act_funcs.append(identity); 
    output_der = identity #elementwise_grad(act_funcs[-1],0);
    act_ders.append(output_der)

    PDE = Diffusion1D(0)
    f = PDE.right_hand_side

    x = anp.linspace(0,1,10)
    t = anp.linspace(0,1,10)
    domain_array = anp.array([x,t])

    network = FFNNetwork(net_in_size, layer_out_sizes,act_funcs,act_ders,PDE,f,dimension=PDE.dimension)
    #network.create_layers()
    P = network.train_network(learn_rate=0.01,epochs=2)
    network.evaluate(P)

    print(anp.max(network.abs_diff))