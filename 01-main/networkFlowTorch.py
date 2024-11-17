from PDEq import *
from support import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import autograd.numpy as anp
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.backend import set_floatx, random_bernoulli
#from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import get
from tensorflow.keras import optimizers, regularizers


class FFNNetworkFlow:

    def __init__(self,
                 layer_output_size,
                 activation_functions,
                 PDE: Functions,
                 source_function,
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
                          init_points=10):
        
        DType = self.DTYPE

        x0,xN = bounds[0]; t0,tN = bounds[1]

        ## Setting domain bounds
        #self.Lb,self.Ub = tf.constant([x0,t0],dtype=DType), tf.constant([xN,tN],dtype=DType)  #x-first
        self.Lb,self.Ub = tf.constant([t0,x0],dtype=DType), tf.constant([tN,xN],dtype=DType)   #t-first
        #print(self.Lb,self.Ub)


        ## Initial boundary data
        #x_0 = tf.random.uniform((init_points,1), self.Lb[0], self.Ub[0], dtype=DType)
        x_0 = tf.random.uniform((init_points,1), self.Lb[1], self.Ub[1], dtype=DType)
        #t_0 = tf.ones((init_points,1),dtype=DType)*self.Lb[1]
        t_0 = tf.ones((init_points,1),dtype=DType)*self.Lb[0]
        #X_0 = tf.concat([x_0,t_0], axis=1)
        X_0 = tf.concat([t_0,x_0], axis=1)
        #print('X0',X_0.shape)

        ## Boundary data
        #x_b = self.Lb[0] + (self.Ub[0] - self.Lb[0]) * random_bernoulli((bound_points,1), 0.5, dtype=DType) #tf.random.uniform((bound_points,1),dtype=DType)
        x_b = self.Lb[1] + (self.Ub[1] - self.Lb[1]) * random_bernoulli((bound_points,1), 0.5, dtype=DType)
        #t_b = tf.random.uniform((bound_points,1), self.Lb[1], self.Ub[1], dtype=DType) 
        t_b = tf.random.uniform((bound_points,1), self.Lb[0], self.Ub[0], dtype=DType) 
        #X_b = tf.concat([x_b,t_b], axis=1)
        X_b = tf.concat([t_b,x_b], axis=1)
        #print('Xb',X_b.shape)
        


        ## Collcation points
        #x_r = tf.random.uniform((colloc_points,1), self.Lb[0], self.Ub[0], dtype=DType)
        x_r = tf.random.uniform((colloc_points,1), self.Lb[1], self.Ub[1], dtype=DType)

        #t_r = tf.random.uniform((colloc_points,1), self.Lb[1], self.Ub[1], dtype=DType)
        t_r = tf.random.uniform((colloc_points,1), self.Lb[0], self.Ub[0], dtype=DType)
        
        #self.X_r = tf.concat([x_r,t_r], axis=1)
        self.X_r = tf.concat([t_r,x_r], axis=1)
        #print('Xr',self.X_r.shape)

        ## Evaluation of initial and boundary conditions for X_0, X_b
        u_0 = self.PDE.init_function(X_0,self.domain)
        u_b = self.PDE.boundary_function(X_b,domain=self.domain)
        #print('u0',u_0.shape)
        #print('ub',u_b.shape)


        '''fig = plt.figure(figsize=(9,6))
        plt.scatter(t_0, x_0, c=u_0, marker='X', vmin=-1, vmax=1)
        plt.scatter(t_b, x_b, c=u_b, marker='p', vmin=-1, vmax=1)
        plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.show()'''

        self.X_data = [X_0,X_b]
        self.u_data = [u_0,u_b]


    def create_layers(self,lmbda):

        num_layers = anp.size(self.layer_out_size)
        dim = self.PDE.dimension

        ## Tensorflow's FFNN - model
        self.model = Sequential()

        ## Input layer, based on dimension of PDE
        self.model.add(Input(shape=[dim,]))

        ## Scaling layer??? (see PINN_Solver.ipynb for details)
        scaling_layer = Lambda(lambda x: 2.0*(x - self.Lb)/(self.Ub - self.Lb) - 1.0)
        self.model.add(scaling_layer)

        ## Hidden layers plus output layer from layers list
        for i in range(num_layers):
            self.model.add(Dense(self.layer_out_size[i],activation=get(self.act_func[i]),
                            kernel_regularizer=regularizers.l2(lmbda)))
            
    def choose_optimizer(self):

        if self.eta is None:
            self.eta = optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
        
        if self.method == 'adam':
            self.optimzer = optimizers.Adam(learning_rate=self.eta)
        elif self.method == 'rmsprop':
            self.optimzer = optimizers.RMSprop(learning_rate=self.eta)
        elif self.method == 'adagrad':
            self.optimzer = optimizers.Adagrad(learning_rate=self.eta)


    def compute_residual(self):
        u_list = []
        with tf.GradientTape(persistent=True) as tape:

            #x,t = self.X_r[:,0:1], self.X_r[:,1:2]  
            t,x = self.X_r[:,0:1], self.X_r[:,1:2]  

            tape.watch(t)
            tape.watch(x)

            #u_list.append(self.model(tf.stack([x[:,0],t[:,0]],axis=1)))
            u_list.append(self.model(tf.stack([t[:,0],x[:,0]],axis=1)))

            u_list.append(tape.gradient(u_list[0], t)) #du_dt
            u_list.append(tape.gradient(u_list[0], x)) #du_dx
        
        u_list.append(tape.gradient(u_list[1], t)) #du_dtt
        u_list.append(tape.gradient(u_list[2], x)) #du_dxx
        
        del tape

        return self.PDE.residual_function(u=u_list)


    def compute_cost(self):
        res = self.compute_residual()
        phi_r = tf.reduce_mean(tf.square(res))

        cost = phi_r

        for i in range(len(self.X_data)):
            u_predition = self.model(self.X_data[i])
            cost += tf.reduce_mean(tf.square(self.u_data[i] - u_predition))
        
        return cost
    
    def compute_gradient(self):
        with tf.GradientTape(persistent=True) as tape:
            
            cost = self.compute_cost()
        
        gradient = tape.gradient(cost, self.model.trainable_variables)

        del tape

        return cost, gradient
    
    def train_network(self,epochs=100):

        @tf.function
        def train_step():
            cost, param_grad = self.compute_gradient()

            self.optimzer.apply_gradients(zip(param_grad, self.model.trainable_variables))

            return cost
        
        self.cost_history = []

        #t0 = time()
        for i in range(epochs):
            cost = train_step()
            self.cost_history.append(cost.numpy())

            if i % 50 == 0:
                print('Iteration: %i: Cost = %1.5e' %(i,cost))

        print('Final cost = %1.5e' %(cost))
        #print('Time: %' %(t0))

    def evaluate(self):
        #x,t = self.domain_array
        t,x = self.domain_array

        Nx,Nt = np.size(x), np.size(t)

        #xx,tt = np.meshgrid(x,t)
        tt,xx = np.meshgrid(t,x)
        #Xgrid = np.vstack([xx.flatten(),tt.flatten()]).T
        Xgrid = np.vstack([tt.flatten(),xx.flatten()]).T


        net_sol = self.model(tf.cast(Xgrid,dtype=self.DTYPE))
        self.network_solution = net_sol.numpy().reshape(Nt,Nx)
        analytic_array = (Xgrid[:,1],Xgrid[:,0]) #u = u(x,t)
        analytic_sol = self.PDE.analytical(domain_array=analytic_array,domain=self.domain)
        self.analytic = analytic_sol.numpy().reshape(Nt,Nx)
        #self.analytic = np.zeros((Nx,Nt))
        #for i, xi in enumerate(x):
        #    for n, tn in enumerate(t):
        #        point = np.array([tn,xi])
        #        self.analytic[i,n] = self.PDE.analytical(point,domain=self.domain)
        
        print(self.network_solution.shape)
        print(self.analytic.shape)

        self.abs_diff = np.abs(self.network_solution - self.analytic)


    def plot_results(self,save=False,f_name='gen_name.png'):
        self.evaluate()
        
        #x,t = self.domain_array
        t,x = self.domain_array

        tt,xx = np.meshgrid(t,x)
        
        #Xgrid = np.vstack([tt.flatten(),xx.flatten()]).T

        #net_sol = self.model(tf.cast(Xgrid,dtype=self.DTYPE))
        #self.network_solution = net_sol.numpy().reshape(Nt,Nx)

        plot2D(tt,xx,self.network_solution,
               labels=['Network solution','t','x','u(x,t)'],
               save=save,f_name=f_name)
        plot2D(tt,xx,self.analytic,
                labels=['Analytical solution','t','x','u(x,t)'],
                save=save,f_name=f_name)
        plot2D(tt,xx,self.abs_diff,
                labels=['Difference','t','x','u(x,t)'],
                save=save,f_name=f_name)
        

class FFNNetworkFlow2D(FFNNetworkFlow):

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
        
        DType = self.DTYPE

        t0,tN = bounds[0]
        x0,xN = bounds[1]
        y0,yN = bounds[2]
        
        ## Setting domain bounds
        self.Lb,self.Ub = tf.constant([t0,x0,y0],dtype=DType), tf.constant([tN,xN,yN],dtype=DType) 

        ## Initial boundary data
        t_0 = tf.ones((init_points,1),dtype=DType)*self.Lb[0]
        x_0 = tf.random.uniform((init_points,1), self.Lb[1], self.Ub[1], dtype=DType)
        y_0 = tf.random.uniform((init_points,1), self.Lb[2], self.Ub[2], dtype=DType)
        X_0 = tf.concat([t_0,x_0,y_0], axis=1)

        ## Boundary data
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
        u_0 = PDE.init_function(X_0,self.domain)
        u_b = PDE.boundary_function(X_b,domain=self.domain)


        self.X_data = [X_0,X_b]
        self.u_data = [u_0,u_b]

    def compute_residual(self):
        u_list = []
        with tf.GradientTape(persistent=True) as tape:

            #t,x = self.X_r[:,0], self.X_r[:,1]
            t,x,y = self.X_r[:,0:1], self.X_r[:,1:2], self.X_r[:,2:3]  

            tape.watch(t); tape.watch(x); tape.watch(y)

            u_list.append(self.model(tf.stack([t[:,0],x[:,0],y[:,0]],axis=1)))
            #print(u == u)
            u_list.append(tape.gradient(u_list[0], t))
            u_list.append(tape.gradient(u_list[0], x))
            u_list.append(tape.gradient(u_list[0], y))

        
        u_list.append(tape.gradient(u_list[1], t))
        u_list.append(tape.gradient(u_list[2], x))
        u_list.append(tape.gradient(u_list[3], x))

        del tape

        return PDE.residual_function(u=u_list)
    
    def evaluate(self):
        t,x,y = self.domain_array
        Nx,Ny,Nt = np.size(x), np.size(y), np.size(t)

        tt,xx,yy = np.meshgrid(t,x,y,indexing='ij') 
        #xx,yy,tt = np.meshgrid(x,y,t) 

        Xgrid = np.vstack([tt.flatten(),xx.flatten(),yy.flatten()]).T
        print(Xgrid[1].shape)

        net_sol = self.model(tf.cast(Xgrid,dtype=self.DTYPE))
        self.network_solution = net_sol.numpy().reshape(Nt,Nx,Ny)

        #analytic_array = (Xgrid[:,1:2],Xgrid[:,2:3],Xgrid[:,0:1])
        analytic_array = (Xgrid[:,1],Xgrid[:,2],Xgrid[:,0])
        #print(analytic_array)
        analytic_sol = self.PDE.analytical(domain_array=analytic_array,domain=self.domain)
        self.analytic = analytic_sol.numpy().reshape(Nt,Nx,Ny)
        
        print(self.network_solution.shape)
        print(self.analytic.shape)

        self.abs_diff = np.abs(self.network_solution - self.analytic)
        

    def plot_results(self, save=False, f_name='gen_name.png'):
        self.evaluate()
        t,x,y = self.domain_array
        #Nx,Ny,Nt = np.size(x), np.size(y), np.size(t)
        tt,xx,yy = np.meshgrid(t,x,y,indexing='ij')
        #xx,yy,tt = np.meshgrid(x,y,t)


        ## Choosing time-instances to plot for
        #idx = [5,int(anp.size(t)/2),anp.size(t)-1]
        idx = [int(anp.size(t)/2)]

        print(idx)
        for i in range(len(idx)):
            plot2D(xx[idx[i],:,:],yy[idx[i],:,:],self.network_solution[:,:,idx[i]],
                   labels=['Network solution','y','x','u(x,y,t)'],
                save=save,f_name=f_name)
            plot2D(xx[idx[i],:,:],yy[idx[i],:,:],self.analytic[:,:,idx[i]],
                   labels=['Analytical solution','y','x','u(x,y,t)'],
                   save=save,f_name=f_name)
            plot2D(xx[idx[i],:,:],yy[idx[i],:,:],self.abs_diff[:,:,idx[i]],
                   labels=['Difference','y','x','u(x,y,t)'],
                   save=save,f_name=f_name)


if __name__ == '__main__':
    problem = '2d'
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
        T0,T,L0,Lx = 0,1,0,1

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
