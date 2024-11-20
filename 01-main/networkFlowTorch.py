from PDEq import *
from support import *

#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR

import tensorflow as tf
#gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import autograd.numpy as anp

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.backend import set_floatx, random_bernoulli
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import get
from tensorflow.keras import optimizers, regularizers


class FFNNetworkFlow:
    """ Class using a Physics Informed Neural Network (PINN) to solve partial differential equations (PDE)'s,
        using **TensorFlow** and the **Keras API**. This base class handles PDE's for one spatial dimension.
        It uses methods implemented in different `PDE`-classes defined in *PDEq.py*"""

    def __init__(self,
                 layer_output_size,
                 activation_functions,
                 PDE: Functions,
                 source_function: Functions.right_hand_side,
                 domain_array,
                 domain=[0,1],
                 gd_method='adam',
                 learning_rate=0.01,
                 random_state = 1,
                 ):
        
        self.layer_out_size = layer_output_size
        self.act_func       = activation_functions
        self.PDE            = PDE
        self.source         = source_function
        self.domain         = domain
        self.domain_array   = domain_array
        self.method         = gd_method
        self.eta            = learning_rate
        self.random_seed    = random_state

        self.model = None

        tf.random.set_seed(self.random_seed)

        self.DTYPE = 'float32'
        set_floatx(self.DTYPE)

        self.choose_optimizer()


    def collocation_setup(self,bounds=([0,1],[0,1]),
                          colloc_points=1000,
                          bound_points=10,
                          init_points=10,
                          plot_colloc=False):
        """ Method setting up the collocation tensor and initial and boundary points for the network
            Spatial boundary points defined using a `random_bernoulli`-method
            
            Results in:
            ---
            Collocation tensor, `X_r`; initial and boundary data, `X_data`; PDE evaluated at initial and 
            boundary points, `u_data`.
        """
        ## Local variable for data-type
        DType = self.DTYPE

        ## Setting domain boundary and limits (lower and upper bounds)
        x0,xN = bounds[0]; t0,tN = bounds[1]
        self.Lb,self.Ub = tf.constant([t0,x0],dtype=DType), tf.constant([tN,xN],dtype=DType)

        ## Initial data points
        x_0 = tf.random.uniform((init_points,1), self.Lb[1], self.Ub[1], dtype=DType)
        t_0 = tf.ones((init_points,1),dtype=DType)*self.Lb[0]
        X_0 = tf.concat([t_0,x_0], axis=1)

        ## Domain boundary points
        x_b = self.Lb[1] + (self.Ub[1] - self.Lb[1]) * random_bernoulli((bound_points,1), 0.5, dtype=DType)
        t_b = tf.random.uniform((bound_points,1), self.Lb[0], self.Ub[0], dtype=DType) 
        X_b = tf.concat([t_b,x_b], axis=1)

        ## Collcation points
        x_r = tf.random.uniform((colloc_points,1), self.Lb[1], self.Ub[1], dtype=DType)
        t_r = tf.random.uniform((colloc_points,1), self.Lb[0], self.Ub[0], dtype=DType)
        self.X_r = tf.concat([t_r,x_r], axis=1)

        ## Evaluation of initial and boundary conditions for X_0, X_b
        u_0 = self.PDE.init_function(X_0,self.domain)
        u_b = self.PDE.boundary_function(X_b,domain=self.domain)

        ## Plot of collocation setup
        if plot_colloc == True:
            fig = plt.figure(figsize=(9,6))
            plt.scatter(t_0, x_0, c=u_0, marker='X', vmin=-1, vmax=1)
            plt.scatter(t_b, x_b, c=u_b, marker='p', vmin=-1, vmax=1)
            plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
            plt.xlabel('$t$')
            plt.ylabel('$x$')
            plt.show()

        ## Storing domain and PDE-data
        self.X_data = [X_0,X_b]
        self.u_data = [u_0,u_b]


    def create_layers(self,lmbda):
        """ Method creating the layers of the network, using the `Sequential`-class from the **Keras API**,
            with `Dense`-layers. 
            
            A scaling layer is added between the input and first hidden layer using
            a *lambda*-function. 
            
            All hidden layers are assigned the same activation function from the class
            initialization, the output layer has no activation
        """

        num_layers = anp.size(self.layer_out_size)
        dim = self.PDE.dimension

        ## Tensorflow's FFNN - model
        self.model = Sequential()

        ## Input layer, based on dimension of PDE
        self.model.add(Input(shape=[dim,]))

        ## Scaling layer (see PINN_Solver.ipynb for details)
        scaling_layer = Lambda(lambda x: 2.0*(x - self.Lb)/(self.Ub - self.Lb) - 1.0)
        self.model.add(scaling_layer)

        ## Hidden layers plus output layer from layers list
        for i in range(num_layers):
            self.model.add(Dense(self.layer_out_size[i],activation=get(self.act_func[i]),
                            kernel_regularizer=regularizers.l2(lmbda)))
            
    def choose_optimizer(self):
        """ Method choosing the scheduler and gradient descent method based on the class initialization."""

        if self.eta is None:
            self.eta = optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
        
        if self.method == 'adam':
            self.optimzer = optimizers.Adam(learning_rate=self.eta)
        elif self.method == 'rmsprop':
            self.optimzer = optimizers.RMSprop(learning_rate=self.eta)
        elif self.method == 'adagrad':
            self.optimzer = optimizers.Adagrad(learning_rate=self.eta)


    def compute_residual(self):
        """ Method assembling a list of gradients to build out the residual from of the
            PDE. Gradinets computed using TensorFlow's `GradientTape`-class
            Derivatives up to second order. 
            
            Returns
            ---
            Residual form of the PDE 
        """

        u_list = []
        with tf.GradientTape(persistent=True) as tape:
            t,x = self.X_r[:,0:1], self.X_r[:,1:2]  

            tape.watch(t)
            tape.watch(x)

            u_list.append(self.model(tf.stack([t[:,0],x[:,0]],axis=1)))

            u_list.append(tape.gradient(u_list[0], t)) #du_dt
            u_list.append(tape.gradient(u_list[0], x)) #du_dx
        
        u_list.append(tape.gradient(u_list[1], t)) #du_dtt
        u_list.append(tape.gradient(u_list[2], x)) #du_dxx
        
        del tape

        return self.PDE.residual_function(u=u_list)


    def compute_cost(self):
        """ Method computing the cost function as a sum of the residual MSE and the 
            MSE from the model evaluated at the boundary and initial points"""
        ## Computing residual and the MSE of the residual
        res = self.compute_residual()
        phi_r = tf.reduce_mean(tf.square(res))

        cost = phi_r

        ## Adding the MSE from evaluating the model at the initial and boundary points
        for i in range(len(self.X_data)):
            u_predition = self.model(self.X_data[i])
            cost += tf.reduce_mean(tf.square(self.u_data[i] - u_predition))
        
        return cost
    
    def compute_gradient(self):
        """ Method that computes the gradient of the cost function using **TensorFlow**'s 
            `GradientTape`-class

            Returns:
            ---
            Cost and gradient variables for backpropagation     
        """
        with tf.GradientTape(persistent=True) as tape:
            
            cost = self.compute_cost()
        
        gradient = tape.gradient(cost, self.model.trainable_variables)

        del tape

        return cost, gradient
    
    def train_network(self,epochs=100, tol=None):
        """ Method that trains the network for a given number of epochs. If a tolerance is given,
            the training is terminated if the error reduction is within the tolerance for 5 training
            steps. 

            Stores the history of the cost-values as class-attribute.    
        """

        @tf.function
        def train_step():
            """ Method computing the gradinet and performing the backpropagation
               for one training step """
            cost, param_grad = self.compute_gradient()

            self.optimzer.apply_gradients(zip(param_grad, self.model.trainable_variables))

            return cost
        
        self.cost_history = []
        
        ## Training loop
        for i in range(epochs):
            cost = train_step()
            self.cost_history.append(cost.numpy())

            if i % int(epochs/10) == 0:
                print('Iteration: %i: Cost = %1.5e' %(i,cost))

        print('Final cost = %1.5e' %(cost))

    def evaluate(self):
        """ Method evaluating the network model against an analytical solution of the PDE.

            Creates instance variables for the final prediction, analytical solution and 
            absolute difference between the two, as well as the R2-score of the prediction?. 
        """
        ## Setting up meshgrid
        t,x = self.domain_array
        Nt,Nx = np.size(t), np.size(x)
        tt,xx = np.meshgrid(t,x)

        ## Array for evaluating the network and solution
        Xgrid = np.vstack([tt.flatten(),xx.flatten()]).T

        ## Evaluating network
        net_sol = self.model(tf.cast(Xgrid,dtype=self.DTYPE))
        self.network_solution = net_sol.numpy().reshape(Nt,Nx)

        ## Computing analytical solution
        analytic_array = (Xgrid[:,1],Xgrid[:,0]) #u = u(x,t)
        analytic_sol = self.PDE.analytical(domain_array=analytic_array,domain=self.domain)
        self.analytic = analytic_sol.numpy().reshape(Nt,Nx)

        ## Calculating the difference between the solutions
        self.abs_diff = np.abs(self.network_solution - self.analytic)

        ## R2-score?


    def plot_results(self,plots='all',
                     idx=[0,-1],
                     save=False,
                     f_name=['gen_name_1','gen_name_2','gen_name_3']
                     ):
        """ Plotting-method that uses the variables from the `evaluate`-method to make 2D-plots 
            the network solution, analytical solution and difference between the two, plotting 
            the entire solution u = u(t,x), for all t, x in a surface-plot.

            It also generates line-plots at defined time-instances, comparing the network solution
            to the analytical solution at those instances.

            Parameters
            ---
            plots : str
                **'all'**: all figures, minimum 4; **'network'**; only network and difference surface-plot; 
                **'exact'**: only analytical solution; **'slices'**: only line-plots of time-instances
            idx : list
                List of time-instances to plot. If **len(idx)** > 3n, n = 1,2,..., multiple 
                figures are made. 
            save : bool
                `True` saves the figures to current directory, using the names provided in **f_name**-list
            f_name : list, str
                List of figure names. 
        """

        ## Evaluating results before plotting
        self.evaluate()

        t,x = self.domain_array
        tt,xx = np.meshgrid(t,x)

        if plots == 'all' or plots == 'network':
            plot2D(tt,xx,self.network_solution,
                labels=['Network solution','t','x','u(t,x)'],
                save=save,f_name=f_name[0])
        if plots == 'all' or plots == 'exact':
            plot2D(tt,xx,self.analytic,
                labels=['Analytical solution','t','x','u(t,x)'],
                save=save,f_name=f_name[1])
        if plots == 'all' or plots == 'network':
            plot2D(tt,xx,self.abs_diff,
                labels=['Difference','t','x','u(t,x)'],
                save=save,f_name=f_name[2])
            
        ## Line plots showing different slices of surface defined by x,t
        if plots == 'slices' or plots == 'all':
            t_id = []; net_sol = []; analytic_res = []; 
            fig,ax = plt.subplots(1,len(idx),figsize=(12,4))
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


class FFNNetworkFlow2D(FFNNetworkFlow):
    """ Class build on the 1D-network class, with modified methods for dealing with PDE that are
        two-dimensional in space """

    def __init__(self,
                 layer_output_size,
                 activation_functions,
                 PDE, 
                 source_function, 
                 domain_array, 
                 domain=([0,1],[0,1]), 
                 gd_method='adam', 
                 learning_rate=0.01, 
                 random_state=1):
        super().__init__(layer_output_size, activation_functions, PDE, source_function, 
                         domain_array, domain, gd_method, learning_rate, random_state)
        
    def collocation_setup(self, bounds=([0,1],[0,1],[0,1]), 
                          colloc_points=1000, 
                          bound_points=10, 
                          init_points=10):
        """ Method setting up the collocation tensor and initial and boundary points for the network
            Spatial boundary points defined using a `random_bernoulli`-method
            
            Results in:
            ---
            Collocation tensor, `X_r`; initial and boundary data, `X_data`; PDE evaluated at initial and 
            boundary points, `u_data`.
        """
        
        ## Local variable for data-type
        DType = self.DTYPE

        ## Setting domain boundary and limits (lower and upper bounds)
        t0,tN = bounds[0]
        x0,xN = bounds[1]
        y0,yN = bounds[2]
        self.Lb,self.Ub = tf.constant([t0,x0,y0],dtype=DType), tf.constant([tN,xN,yN],dtype=DType) 

        ## Initial data points
        t_0 = tf.ones((init_points,1),dtype=DType)*self.Lb[0]
        x_0 = tf.random.uniform((init_points,1), self.Lb[1], self.Ub[1], dtype=DType)
        y_0 = tf.random.uniform((init_points,1), self.Lb[2], self.Ub[2], dtype=DType)
        X_0 = tf.concat([t_0,x_0,y_0], axis=1)

        ## Domain boundary points
        t_b = tf.random.uniform((bound_points,1), self.Lb[0], self.Ub[0], dtype=DType)
        x_b = self.Lb[1]+(self.Ub[1]-self.Lb[1]) * random_bernoulli((bound_points,1), 0.5, dtype=DType)
        y_b = self.Lb[2]+(self.Ub[2]-self.Lb[2]) * random_bernoulli((bound_points,1), 0.5, dtype=DType) 
        X_b = tf.concat([t_b,x_b,y_b], axis=1)

        ## Collcation points
        t_r = tf.random.uniform((colloc_points,1), self.Lb[0], self.Ub[0], dtype=DType)
        x_r = tf.random.uniform((colloc_points,1), self.Lb[1], self.Ub[1], dtype=DType)
        y_r = tf.random.uniform((colloc_points,1), self.Lb[2], self.Ub[2], dtype=DType)
        self.X_r = tf.concat([t_r,x_r,y_r], axis=1)

        ## Evaluation of initial and boundary conditions for X_0, X_b
        u_0 = self.PDE.init_function(X_0,self.domain)
        u_b = self.PDE.boundary_function(X_b,domain=self.domain)

        ## Storing domain and PDE-data
        self.X_data = [X_0,X_b]
        self.u_data = [u_0,u_b]

    def compute_residual(self):
        """ Method assembling a list of gradients to build out the residual from of the
            PDE. Gradinets computed using TensorFlow's `GradientTape`-class. 
            Derivatives up to second order. 
            
            Returns
            ---
            Residual form of the PDE     
        """
        u_list = []
        with tf.GradientTape(persistent=True) as tape:
            ## Local variables of the collocation points
            t,x,y = self.X_r[:,0:1], self.X_r[:,1:2], self.X_r[:,2:3]  

            ## 
            tape.watch(t); tape.watch(x); tape.watch(y)

            ## Computing network u-values
            u_list.append(self.model(tf.stack([t[:,0],x[:,0],y[:,0]],axis=1)))

            ## First order derivatives for the residual
            u_list.append(tape.gradient(u_list[0], t)) #dudt
            u_list.append(tape.gradient(u_list[0], x)) #dudx
            u_list.append(tape.gradient(u_list[0], y)) #dudy

        ## Second order derivatives for the residual
        u_list.append(tape.gradient(u_list[1], t)) #dudtt
        u_list.append(tape.gradient(u_list[2], x)) #dudxx
        u_list.append(tape.gradient(u_list[3], x)) #dudyy

        del tape

        return self.PDE.residual_function(u=u_list)
    
    def evaluate(self):
        """ Method evaluating the network model against an analytical solution of the PDE.

            Creates instance variables for the final prediction, analytical solution and 
            absolute difference between the two, as well as the R2-score of the prediction?. 
        """
        
        ## Setting up meshgrid
        t,x,y = self.domain_array
        Nx,Ny,Nt = np.size(x), np.size(y), np.size(t)
        tt,xx,yy = np.meshgrid(t,x,y,indexing='ij') 

        ## Array for evaluating the network and solution
        Xgrid = np.vstack([tt.flatten(),xx.flatten(),yy.flatten()]).T

        ## Evaluating network
        net_sol = self.model(tf.cast(Xgrid,dtype=self.DTYPE))
        self.network_solution = net_sol.numpy().reshape(Nt,Nx,Ny)

        ## Computing analytical solution
        analytic_array = (Xgrid[:,1],Xgrid[:,2],Xgrid[:,0])
        analytic_sol = self.PDE.analytical(domain_array=analytic_array,domain=self.domain)
        self.analytic = analytic_sol.numpy().reshape(Nt,Nx,Ny)

        ## Calculating the difference between the solutions
        self.abs_diff = np.abs(self.network_solution - self.analytic)
        

    def plot_results(self,
                    plots='all',
                    time_idx=[5,-1],
                    space_idx=([1,5],[1,5]),
                    save=False, f_name='gen_name.png'
                    ):
        """ Plotting-method that uses the variables from the `evaluate`-method to make 2D-plots 
            the network solution, analytical solution and difference between the two, plotting 
            the solution u = u(t,x,y), for all x, y in a surface-plot at given time-instances.

            It also generates line-plots at defined time-instances, comparing the network solution
            to the analytical solution at those instances, for the slices in x and y provided through
            the **space_idx**-input.

            Parameters
            ---
            plots : str
                **'all'**: all figures, minimum 4; **'network'**; only network and difference surface-plot; 
                **'exact'**: only analytical solution; **'slices'**: only line-plots of time-instances
            time_idx : list
                List of time-instances to plot. If **len(idx)** > 3n, n = 1,2,..., multiple 
                figures are made. 
            space_idx : tuple
                Tuple of x and y-values to take slices from. Time-instance provided in **time_idx**
            save : bool
                `True` saves the figures to current directory, using the names provided in **f_name**-list
            f_name : list, str
                List of figure names. 
        """

        self.evaluate()
        t,x,y = self.domain_array
        tt,xx,yy = np.meshgrid(t,x,y,indexing='ij')

        for i in range(len(time_idx)):
            if plots == 'all' or plots == 'finite':
                plot2D(xx[time_idx[i],:,:],yy[time_idx[i],:,:],self.network_solution[:,:,time_idx[i]],
                    labels=['Network solution','y','x','u(x,y,t)'],
                    save=save,f_name=f_name)
            if plots == 'all' or plots == 'exact':
                plot2D(xx[time_idx[i],:,:],yy[time_idx[i],:,:],self.analytic[:,:,time_idx[i]],
                    labels=['Analytical solution','y','x','u(x,y,t)'],
                    save=save,f_name=f_name)
            if plots == 'all' or plots == 'finite':
                plot2D(xx[time_idx[i],:,:],yy[time_idx[i],:,:],self.abs_diff[:,:,time_idx[i]],
                    labels=['Difference','y','x','u(x,y,t)'],
                    save=save,f_name=f_name)
        if plots == 'slices':
            for i in range(len(time_idx)):
                for id in space_idx:
                    raise NotImplementedError


if __name__ == '__main__':
    problem = '1d'
    if problem == '1d':
        tf.random.set_seed(1)

        layer_out_sizes = [20,20,20,20,20,20,20,20,1]

        hidden_func = 'gelu' # sigmoid, relu, elu, leaky_relu, tanh, swish, gelu, hard_sigmoid, exponential

        act_funcs = []; act_ders = []
        for i in range(len(layer_out_sizes)-1):
            act_funcs.append(hidden_func)

        act_funcs.append(None); 

        #PDE = Diffusion1D(sim_type='flow')
        #PDE = Burger1D(sim_type='flow')
        PDE = Wave1D(sim_type='flow')
        f = PDE.right_hand_side

        Nt,Nx = 100,100
        T0,T,L0,Lx = 0,1,-1,1

        x_bound = [L0,Lx]; t_lim = [T0,T]
        x = np.linspace(x_bound[0],x_bound[1],Nx)
        t = np.linspace(t_lim[0],t_lim[1],Nt)
        
        domain_array = anp.array([t,x])
        lmbda = 1e-6
        
        network = FFNNetworkFlow(layer_output_size=layer_out_sizes,
                                activation_functions=act_funcs,
                                PDE=PDE,
                                source_function=f,
                                domain_array=domain_array,
                                domain=x_bound,
                                gd_method='adam',
                                learning_rate=None)
        
        network.collocation_setup(bounds=(x_bound,t_lim),colloc_points=10000,
                                bound_points=50,init_points=50)
        network.create_layers(lmbda=lmbda)
        network.choose_optimizer()
        
        network.train_network(1000)
        #network.evaluate()
        network.plot_results()
        print(np.max(network.abs_diff))
        plt.show()

    elif problem == '2d':
        tf.random.set_seed(1)

        layer_out_sizes = [20,20,20,20,20,20,20,20,1]

        hidden_func = 'gelu' # sigmoid, relu, elu, leaky_relu, tanh, swish, gelu, hard_sigmoid, exponential

        act_funcs = []; act_ders = []
        for i in range(len(layer_out_sizes)-1):
            act_funcs.append(hidden_func)

        act_funcs.append(None); 

        PDE = Diffusion2D(sim_type='flow')
        #PDE = Wave2D(sim_type='flow',m=[2,2])

        f = PDE.right_hand_side

        Nt,Nx,Ny = 100,100,100
        t0,x0,y0 = 0,0,0
        T,Lx,Ly = 1,1,1
        t_lim = [t0,T]; x_bound = [x0,Lx]; y_bound = [y0,Ly]
        t = np.linspace(t_lim[0],t_lim[1],Nt)
        x = np.linspace(x_bound[0],x_bound[1],Nx)
        y = np.linspace(y_bound[0],y_bound[1],Ny)
        
        domain_array = anp.array([t,x,y])
        lmbda = 1e-2
        
        network = FFNNetworkFlow2D(layer_output_size=layer_out_sizes,
                                    activation_functions=act_funcs,
                                    PDE=PDE,
                                    source_function=f,
                                    domain_array=domain_array,
                                    domain=(x_bound,y_bound),
                                    gd_method='adam',
                                    learning_rate=None)
        
        network.collocation_setup(bounds=(x_bound,y_bound,t_lim),
                                  colloc_points=10000,
                                  bound_points=50, 
                                  init_points=50)
        network.create_layers(lmbda=lmbda)
        
        network.train_network(1000)

        network.plot_results()
        print(np.max(network.abs_diff))
        
        plt.show()
