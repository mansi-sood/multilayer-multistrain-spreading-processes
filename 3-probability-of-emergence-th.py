import numpy as np
import scipy
from scipy.special import comb
from scipy import optimize

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
	return g_f(zf,p_0_f)*G_w(zw,p_0_w) 
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

###### CHANGE PARAMETERS HERE ######
tf1 = 0.6 # transmissibility of strain-1 in layer-f
tf2 = 0.8 # transmissibility of strain-2 in layer-f
tw1 = 0.7 # transmissibility of strain-1 in layer-w
tw2 = 0.9 # transmissibility of strain-2 in layer-w

mf1 = 0.1 # probability that strain-1 doesnt undergo mutation in layer-f ($\mu^f_{1,1}$)
mf2 = 0.95 # probability that strain-2 doesnt undergo mutation in layer-f ($\mu^f_{2,2}$)
mw1 = 0.1 # probability that strain-1 doesnt undergo mutation in layer-w 
mw2 = 0.95 # probability that strain-2 doesnt undergo mutation in layer-w

## the degree distribution of each layer a \in {f,w} is a mixture of 0 (w.p. p_0_a) and Poisson(poisson_mean_a) (w.p. 1-p_0_a)
poisson_mean_f  =  1 
poisson_mean_w = 1
p_0_f = 0.0 
p_0_w = 0.0 
####################################

# compute mean and excess mean degree

lambda_f = (1-p_0_f)*poisson_mean_f
lambda_w = (1-p_0_w)*poisson_mean_w
beta_f = poisson_mean_f
beta_w = poisson_mean_w

J = np.array(
[[tf1*mf1*beta_f, tf1*(1-mf1)*beta_f, tw1*mw1*lambda_w, tw1*(1-mw1)*lambda_w],
[tf2*(1-mf2)*beta_f, tf2*mf2*beta_f, tw2*(1-mw2)*lambda_w, tw2*mw2*lambda_w],
[tf1*mf1*lambda_f, tf1*(1-mf1)*lambda_f, tw1*mw1*beta_w, tw1*(1-mw1)*beta_w],
[tf2*(1-mf2)*lambda_f, tf2*mf2*lambda_f, tw2*(1-mw2)*beta_w, tw2*mw2*beta_w]])    

# computing spectral radius of the Jacobian matrix
rho=max(np.linalg.eigvals(J))
print('rho= ',rho)

# computing the probability of emergence
if rho <= 1:
    p = 0
    
else:    
    s = [0.5,0.5,0.5,0.5] #initial guess   
    q = scipy.optimize.root(fixedpt_prob_emergence, s, tol=1e-10).x
    p = 1-gamma1(q[0],q[1],q[2],q[3])
print('prob_emerg= ',p)

# print result
print(poisson_mean_f,poisson_mean_w,tf1,tf2,tw1,tw2,mf1,mf2,p_0_f,p_0_w,p,rho,poisson_mean_f*(1-p_0_f)+poisson_mean_w*(1-p_0_w))






'''Note: for power law degree distribution with cutoff, please update the PGFs as below :
# def lambda_beta_return(alpha, exp_param):
#     denom = float(polylog(alpha, math.exp(-1.0 / exp_param)))
#     lambda_val = float(polylog(alpha-1, math.exp(-1.0 / exp_param))) /denom #first moment
#     beta_val =  float(polylog(alpha-2, math.exp(-1.0 / exp_param))) / denom #second moment, not used later
#     return denom,lambda_val, beta_val

# zeta_f, lambda_f, beta_f = lambda_beta_return(alpha_f, exp_param_f) 
# zeta_w, lambda_w, beta_w = lambda_beta_return(alpha_w, exp_param_w) 
# beta_f = beta_f/lambda_f  - 1
# beta_w = beta_w/lambda_w  - 1

# def g_1(z, alpha, exp_param): # This is the generating function of degree distribution drawn from power law with exp cutoff
#     temp= float(polylog(alpha, z*math.exp(-1.0 / exp_param))) 
#     if alpha == alpha_f:
#         return temp/(zeta_f)
#     else:
#         return temp/(zeta_w)

# def G_1(z, alpha, exp_param): # This is the generating function of EXCESS degree distribution when degree distribution is power law with exp cutoff
#     temp = float(polylog(alpha-1, z*math.exp(-1.0 / exp_param)))
#     if alpha == alpha_f:

#         return temp/(z*lambda_f*zeta_f)
#     else:
#         return temp/(z*lambda_w*zeta_w)

## Ensure that the datatype for the Jacobian matrix is correctly updated J_float = np.asarray(J, dtype = np.float64, order ='C')'''

