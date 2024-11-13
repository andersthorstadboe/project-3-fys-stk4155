from network import *
from PDEq import *
from support import *

import autograd.numpy as anp
from autograd import elementwise_grad

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

## Random seed
default_seed = 1; anp.random.seed(default_seed)

## Figure defaults
plt.rcParams["figure.figsize"] = (8,3); plt.rcParams["font.size"] = 10

anp.random.seed(default_seed)

problem_dimension = 2

net_in_size = 10
layer_out_sizes = [14,3,15,10,1]

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
f = 0

x = anp.linspace(0,1,10)
t = anp.linspace(0,1,10)
domain_array = anp.array([x,t])

network = FFNNetwork(net_in_size, layer_out_sizes,act_funcs,act_ders,PDE,f,2)
#network.create_layers()
P = network.train_network(learn_rate=0.01,epochs=1)
