import numpy as np
import scipy
from scipy.special import comb
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"

# Define generating functions
def g_f(z,p_0_f): # This is the generating function of degree distribution for layer-f
    return p_0_f + (1-p_0_f)*np.exp(poisson_mean_f*(z-1))
def G_f(z,p_0_f): # This is the generating function of EXCESS degree distribution for layer-f
    if p_0_f == 1:
        return 1
    return np.exp(poisson_mean_f*(z-1))
def g_w(z,p_0_w ): # This is the generating function of degree distribution for layer-w
    return p_0_w + (1-p_0_w)*np.exp(poisson_mean_w*(z-1))
def G_w(z,p_0_w): # This is the generating function of EXCESS degree distribution for layer-w
    if p_0_w == 1:
        return 1
    return np.exp(poisson_mean_w*(z-1))

# Below are the definitions for joint PGFs as defined in the paper
def g(zf,zw):
	return g_f(zf,p_0_f)*g_w(zw,p_0_w )
def Gf(zf,zw):
	return G_f(zf,p_0_f)*g_w(zw,p_0_w ) 
def Gw(zf,zw):
	return g_f(zf,p_0_f)*G_w(zw, p_0_w) 
def gamma1(zf1,zf2,zw1,zw2): 
    return g((1-tf1)+(tf1)*(mf1*zf1+(1-mf1)*zf2),(1-tw1)+(tw1)*(mw1*zw1+(1-mw1)*zw2)) 
def Gamma1f(zf1,zf2,zw1,zw2): 
    return Gf((1-tf1)+(tf1)*(mf1*zf1+(1-mf1)*zf2),(1-tw1)+(tw1)*(mw1*zw1+(1-mw1)*zw2))
def Gamma2f(zf1,zf2,zw1,zw2): 
    return Gf((1-tf2)+(tf2)*((1-mf2)*zf1+(mf2)*zf2),(1-tw2)+(tw2)*((1-mw2)*zw1+(mw2)*zw2))
def Gamma1w(zf1,zf2,zw1,zw2): 
    return Gw((1-tf1)+(tf1)*(mf1*zf1+(1-mf1)*zf2),(1-tw1)+(tw1)*(mw1*zw1+(1-mw1)*zw2))
def Gamma2w(zf1,zf2,zw1,zw2):
    return Gw((1-tf2)+(tf2)*((1-mf2)*zf1+(mf2)*zf2),(1-tw2)+(tw2)*((1-mw2)*zw1+(mw2)*zw2))

#  The four recursive fixed point equations for qf1, qf2 qw1, qw2
def fixedpt_prob_emergence(x):
    return [Gamma1f(x[0],x[1],x[2],x[3])-x[0],Gamma2f(x[0],x[1],x[2],x[3])-x[1],Gamma1w(x[0],x[1],x[2],x[3])-x[2],Gamma2w(x[0],x[1],x[2],x[3])-x[3]]

##-----Parameters that characterize the spreading process-----##

###### CHANGE PARAMETERS HERE ######
tf1 = 0.4 # transmissibility of strain-1 in layer-f
tf2 = 0.8 # transmissibility of strain-2 in layer-f
c = 1.0 # the ratio tw1/tf1 and tw2/tf2
####################################

tw1 = c*tf1
tw2 = c*tf2
knot = 1e-10
mf2_0 = 1-knot # mf2_0 is the initial value of mf2 which gives the probability that strain-2 doesnt undergo mutation $\mu_{2,2}$
mf1_0 = 1.0 # mf1_0 is the initial value of mf1 which gives the probability that strain-1 doesnt undergo mutation $\mu_{1,1}$

##-----Parameters that characterize the degree distribution of each layer-----##
## the degree distribution of each layer is a mixture of 0 and Poisson distribution
p_0_w_0 = 0 # p_0_w_0 is the initial value of p_0_w which gives the probability that a node in layer-w is drawn from 0 distribution
p_0_f_0 = 0 # p_0_f_0 is the initial value of p_0_f which gives the probability that a node in layer-f is drawn from 0 distribution 
poisson_mean_f_0 = 1.2 # with probability (1-p_0_f), the degree distribution in layer-f is Poisson(poisson_mean_f); poisson_mean_f_0 is the initial value of poisson_mean_f
poisson_mean_w_0 = 1.2  # with probability (1-p_0_w), the degree distribution in layer-w is Poisson(poisson_mean_w); poisson_mean_w_0 is the initial value of poisson_mean_w


# Initialize mean and excess mean degree
poisson_mean_f = poisson_mean_f_0 
poisson_mean_w = poisson_mean_w_0
p_0_f = p_0_f_0
p_0_w = p_0_w_0
lambda_f = (1-p_0_f)*poisson_mean_f
lambda_w = (1-p_0_w)*poisson_mean_w
beta_f = poisson_mean_f
beta_w = poisson_mean_w

# Set plot properties
gridlen = 11
end_alpha = 0.99

output_matrix = np.zeros((gridlen, gridlen))
i=0
for p_0_f in np.linspace(0.0,end_alpha,gridlen):
    j=0
    for mf1 in np.linspace(0.0,end_alpha,gridlen):

        mf2 = mf2_0
        mw1 = mf1
        mw2 = mf2

        lambda_f = (1-p_0_f)*poisson_mean_f
        lambda_w = (1-p_0_w)*poisson_mean_w
        beta_f = poisson_mean_f
        beta_w = poisson_mean_w

        J = np.array(
        [[tf1*mf1*beta_f, tf1*(1-mf1)*beta_f, tw1*mw1*lambda_w, tw1*(1-mw1)*lambda_w],
        [tf2*(1-mf2)*beta_f, tf2*mf2*beta_f, tw2*(1-mw2)*lambda_w, tw2*mw2*lambda_w],
        [tf1*mf1*lambda_f, tf1*(1-mf1)*lambda_f, tw1*mw1*beta_w, tw1*(1-mw1)*beta_w],
        [tf2*(1-mf2)*lambda_f, tf2*mf2*lambda_f, tw2*(1-mw2)*beta_w, tw2*mw2*beta_w]])    
        w,v  = np.linalg.eig(J)

        # Find spectral radius of J matrix, denoted as rho
        rho=max(np.linalg.eigvals(J))
        #print('rho= ',rho)

        # Find the probability of emergence, denoted as p
        if rho <= 1:
            #print('prob_emerg',0)
            p = 0
            
        else:    
            s = [0.5,0.5,0.5,0.5] #initialization  
            q = scipy.optimize.root(fixedpt_prob_emergence, s).x
            p = 1-gamma1(q[0],q[1],q[2],q[3])
            #print('prob_emerg',p)

        output_matrix[i,j]=p
        print(p_0_f,mf1,output_matrix[i,j],rho)
        j+=1
    i+=1


fig, ax = plt.subplots()
im = ax.imshow(output_matrix, cmap='viridis')

# Set ticks and labels
ax.set_xticks(np.arange(len(np.linspace(0.0, 1.0, gridlen))))
ax.set_yticks(np.arange(len(np.linspace(0.0, 1.0, gridlen))))
ax.set_xticklabels(np.round(np.linspace(0.0, 1.0, gridlen), 2), fontsize=18, fontweight='bold')
ax.set_yticklabels(np.round(np.linspace(0.0, 1.0, gridlen), 2), fontsize=18, fontweight='bold')

# Set axis labels
ax.set_xlabel('$\mu_{11}$', fontsize=32,fontweight='bold')
ax.set_ylabel('$\\alpha_f$', fontsize=32, fontweight='bold', rotation=90)

# Rotate x tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# # Set axis limits
ax.set_xlim(0, len(np.linspace(0.0, end_alpha, gridlen)) - 1)
ax.set_ylim(0, len(np.linspace(0.0, end_alpha, gridlen)) - 1)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=18)

# Set title with increased gap (pad)
ax.set_title(f'$c = {c}, T^f_1 = {tf1}, T^f_2 = {tf2}$', fontsize=20, pad=15)

# Adjust the margins to prevent axis labels from being cropped
plt.subplots_adjust(left=0.15, bottom=0.20, right=0.85, top=0.85)

# Save and show plot
plt.savefig(f"{c}_{tf1}_{tf2}.png", dpi=300)
plt.show()


