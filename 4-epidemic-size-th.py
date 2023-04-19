from __future__ import division
import argparse
import math
import sys, site, os
import time


import numpy as np
from scipy.optimize import fsolve
from scipy.misc import comb  
from scipy.stats import poisson
import scipy.optimize
import scipy.misc

max_degree = 12 # the max index approximation taken in infinite summations 

# Inputs
def parseArgs(args):
    parser = argparse.ArgumentParser(description = 'Parameters')
    parser.add_argument('-lambda_f', type = float, default = 1, help='3 (default); the poisson mean parameter of network F')
    parser.add_argument('-lambda_w', type = float, default = 1 , help='2 (default); the poisson mean parameter of network W')
    parser.add_argument('-tf1', type = float, default = 0.6, help='0.2 (default); the transmissibility of strain-1 on network F')
    parser.add_argument('-tf2', type = float, default = 0.8, help='0.5 (default); the transmissibility of strain-2 on network F')
    parser.add_argument('-tw1', type = float, default = 0.7, help='0.2 (default); the transmissibility of strain-1 on network W')
    parser.add_argument('-tw2', type = float, default = 0.9, help='0.5 (default); the transmissibility of strain-2 on network W')
    parser.add_argument('-mf1', type = float, default = 0.1, help='0.75 (default); the mutation probability from 1 to 1 on network F')
    parser.add_argument('-mf2', type = float, default = 0.95, help='0.75 (default); the mutation probability from 2 to 2 on network F')
    parser.add_argument('-mw1', type = float, default = 0.1, help='0.75 (default); the mutation probability from 1 to 1 on network W')
    parser.add_argument('-mw2', type = float, default = 0.95, help='0.75 (default); the mutation probability from 2 to 2 on network F')
    parser.add_argument('-numCores', type = int, default = 12, help='number of Cores')
    return parser.parse_args(args)




paras = parseArgs(sys.argv[1:])
lambda_f = paras.lambda_f
lambda_w = paras.lambda_w
tf1 = paras.tf1
tf2 = paras.tf2
muf11 = paras.mf1
muf12 = 1 - muf11
muf22 = paras.mf2
muf21 = 1 - muf22
tw1 = paras.tw1
tw2 = paras.tw2
muw11 = paras.mw1
muw12 = 1 - muw11
muw22 = paras.mw2
muw21 = 1 - muw22

# Functions for qf1inf, qf2inf, qw1inf, qw2inf
def obtain_val_qfinf(qf1, qf2, qw1, qw2):
    #print('obtain_val_qfinf called')
    qf1val = 0
    qf2val = 0
    # loop over outer-most summation
    for d_f in xrange(0, max_degree):
        if d_f == 0: continue
        for d_w in xrange(0, max_degree):
            prob_f = poisson.pmf(d_f, lambda_f)
            prob_w = poisson.pmf(d_w, lambda_w)
            prob =  prob_f*prob_w

            middle_sum1 = 0
            middle_sum2 = 0
            # loop over middle summation
            for kf1 in xrange(0, d_f):
                for kw1 in xrange(0, d_w + 1):
                    for kf2 in xrange(0, d_f - kf1):
                        for kw2 in xrange(0, d_w + 1 - kw1):
                            if kf1 == 0 and kf2 == 0 and kw1 == 0 and kw2 == 0: continue

                            inner_sum_1 = 0
                            inner_sum_2 = 0
                            # loop over inner-most summation
                            for xf1 in xrange(0, kf1 + 1):
                                for xw1 in xrange(0 , kw1 + 1):
                                    for xf2 in xrange(0 , kf2 + 1):
                                        for xw2 in xrange(0 , kw2 + 1):
                                            # append inner-most sum
                                            tmp_innerterm = comb(kf1, xf1) * comb (kf2, xf2) * comb(kw1, xw1) * comb (kw2, xw2) *\
                                            tf1**xf1 * tf2**xf2 * (1-tf1)**(kf1-xf1) * (1-tf2)**(kf2-xf2) *\
                                            tw1**xw1 * tw2**xw2 * (1-tw1)**(kw1-xw1) * (1-tw2)**(kw2-xw2)
                                            if xf1+xf2+xw1+xw2 > 0:
                                                inner_sum_1 += tmp_innerterm * ((xf1*muf11 + xf2*muf21 + xw1*muw11 +  xw2*muw21)/(xf1 + xf2 + xw1 + xw2))
                                                inner_sum_2 += tmp_innerterm * ((xf1*muf12 + xf2*muf22 + xw1*muw12 +  xw2*muw22)/(xf1 + xf2 + xw1 + xw2))
                                                #print('iter d_f,d_w,kf1,kf2,kw1,kw2,xf1,xf2,xw1,xw2',d_f,d_w,kf1,kf2,kw1,kw2,xf1,xf2,xw1,xw2)


                            # append middle sum
                            tmp_middleterm = comb(d_f - 1, kf1) * comb(d_f -1 - kf1, kf2) * comb(d_w , kw1) * comb(d_w - kw1, kw2)*\
                            (qf1 ** kf1) * (qf2 ** kf2) * ((1 - qf1 - qf2) ** (d_f - 1 - kf1 - kf2)) *\
                            (qw1 ** kw1) * (qw2 ** kw2) * ((1 - qw1 - qw2) ** (d_w - kw1 - kw2))

                            middle_sum1 +=  tmp_middleterm * inner_sum_1
                            middle_sum2 +=  tmp_middleterm * inner_sum_2


            # append outer-most sum
            qf1val += d_f*prob*1.0/lambda_f * middle_sum1
            qf2val += d_f*prob*1.0/lambda_f * middle_sum2
    return qf1val, qf2val

def obtain_val_qwinf(qf1, qf2, qw1, qw2):
    #print('obtain_val_qwinf called')
    qw1val = 0
    qw2val = 0
    # loop over outer-most summation
    for d_f in xrange(0, max_degree):
        for d_w in xrange(0, max_degree):
            if d_w == 0: continue
            prob_f = poisson.pmf(d_f, lambda_f)
            prob_w = poisson.pmf(d_w, lambda_w)
            prob =  prob_f*prob_w

            middle_sum1 = 0
            middle_sum2 = 0
            # loop over middle summation
            for kf1 in xrange(0, d_f + 1):
                for kw1 in xrange(0, d_w):
                    for kf2 in xrange(0, d_f + 1 - kf1):
                        for kw2 in xrange(0, d_w - kw1):
                            if kf1 == 0 and kf2 == 0 and kw1 == 0 and kw2 == 0: continue

                            inner_sum_1 = 0
                            inner_sum_2 = 0
                            # loop over inner-most summation
                            for xf1 in xrange(0, kf1 + 1):
                                for xw1 in xrange(0 , kw1 + 1):
                                    for xf2 in xrange(0 , kf2 + 1):
                                        for xw2 in xrange(0 , kw2 + 1):
                                            # append inner-most sum
                                            tmp_innerterm = comb(kf1, xf1) * comb (kf2, xf2) * comb(kw1, xw1) * comb (kw2, xw2) *\
                                            tf1**xf1 * tf2**xf2 * (1-tf1)**(kf1-xf1) * (1-tf2)**(kf2-xf2) *\
                                            tw1**xw1 * tw2**xw2 * (1-tw1)**(kw1-xw1) * (1-tw2)**(kw2-xw2)
                                            if xf1+xf2+xw1+xw2 > 0:
                                                inner_sum_1 += tmp_innerterm * ((xf1*muf11 + xf2*muf21 + xw1*muw11 +  xw2*muw21)/(xf1 + xf2 + xw1 + xw2))
                                                inner_sum_2 += tmp_innerterm * ((xf1*muf12 + xf2*muf22 + xw1*muw12 +  xw2*muw22)/(xf1 + xf2 + xw1 + xw2))
                                                #print('iter d_f,d_w,kf1,kf2,kw1,kw2,xf1,xf2,xw1,xw2',d_f,d_w,kf1,kf2,kw1,kw2,xf1,xf2,xw1,xw2)


                            # append middle sum
                            tmp_middleterm = comb(d_f , kf1) * comb(d_f - kf1, kf2) * comb(d_w - 1, kw1) * comb(d_w -1 - kw1, kw2)*\
                            (qf1 ** kf1) * (qf2 ** kf2) * ((1 - qf1 - qf2) ** (d_f - kf1 - kf2)) *\
                            (qw1 ** kw1) * (qw2 ** kw2) * ((1 - qw1 - qw2) ** (d_w - 1 - kw1 - kw2))

                            middle_sum1 +=  tmp_middleterm * inner_sum_1
                            middle_sum2 +=  tmp_middleterm * inner_sum_2


            # append outer-most sum
            qw1val += d_w*prob*1.0/lambda_w * middle_sum1
            qw2val += d_w*prob*1.0/lambda_w * middle_sum2
    return qw1val, qw2val


# Obtaining probability of being infected with strain 1/2 for root node
def obtain_val_Q(qf1, qf2, qw1, qw2):
    #print('obtain_val_Q  called')
    Q = 0
    Q1 = 0
    Q2 = 0
    # loop over outer-most summation
    for d_f in xrange(0, max_degree):
        for d_w in xrange(0, max_degree):
            prob_f = poisson.pmf(d_f, lambda_f)
            prob_w = poisson.pmf(d_w, lambda_w)
            prob =  prob_f*prob_w


            middle_sum1 = 0
            middle_sum2 = 0
            # loop over middle summation
            for kf1 in xrange(0, d_f + 1):
                for kw1 in xrange(0, d_w + 1):
                    for kf2 in xrange(0, d_f + 1 - kf1):
                        for kw2 in xrange(0, d_w + 1 - kw1):


                            inner_sum_1 = 0
                            inner_sum_2 = 0
                            # loop over inner-most summation
                            for xf1 in xrange(0, kf1 + 1):
                                for xw1 in xrange(0 , kw1 + 1):
                                    for xf2 in xrange(0 , kf2 + 1):
                                        for xw2 in xrange(0 , kw2 + 1):
                                            # append inner-most sum
                                            tmp_innerterm = comb(kf1, xf1) * comb (kf2, xf2) * comb(kw1, xw1) * comb (kw2, xw2) *\
                                            tf1**xf1 * tf2**xf2 * (1-tf1)**(kf1-xf1) * (1-tf2)**(kf2-xf2) *\
                                            tw1**xw1 * tw2**xw2 * (1-tw1)**(kw1-xw1) * (1-tw2)**(kw2-xw2)
                                            if xf1+xf2+xw1+xw2 > 0:
                                                inner_sum_1 += tmp_innerterm * ((xf1*muf11 + xf2*muf21 + xw1*muw11 +  xw2*muw21)/(xf1 + xf2 + xw1 + xw2))
                                                inner_sum_2 += tmp_innerterm * ((xf1*muf12 + xf2*muf22 + xw1*muw12 +  xw2*muw22)/(xf1 + xf2 + xw1 + xw2))
                                                #print('iter d_f,d_w,kf1,kf2,kw1,kw2,xf1,xf2,xw1,xw2',d_f,d_w,kf1,kf2,kw1,kw2,xf1,xf2,xw1,xw2)


                            # append middle sum
                            tmp_middleterm = comb(d_f , kf1) * comb(d_f - kf1, kf2) * comb(d_w , kw1) * comb(d_w - kw1, kw2) *\
                            (qf1 ** kf1) * (qf2 ** kf2) * ((1 - qf1 - qf2) ** (d_f - kf1 - kf2)) *\
                            (qw1 ** kw1) * (qw2 ** kw2) * ((1 - qw1 - qw2) ** (d_w - kw1 - kw2))
                            middle_sum1 +=  tmp_middleterm * inner_sum_1
                            middle_sum2 +=  tmp_middleterm * inner_sum_2


            # append outer-most sum
            Q += prob * (middle_sum1 + middle_sum2)
            Q1 += prob * middle_sum1
            Q2 += prob * middle_sum2
    return Q, Q1, Q2

# The four recursive fixed point equations for qf1inf, qf2inf, qw1inf, qw2inf

def equations(s):
    equations.counter+=1
    print('fixed point solver going through iteration # ',equations.counter)
    qf1inf, qf2inf, qw1inf, qw2inf = s
    start1 = time.time()
    qf1inf_val, qf2inf_val = obtain_val_qfinf(qf1inf, qf2inf, qw1inf, qw2inf)
    print("one call of qfi",time.time() - start1)
    start2 = time.time()
    qw1inf_val, qw2inf_val = obtain_val_qwinf(qf1inf, qf2inf, qw1inf, qw2inf)
    print("one call of qwi",time.time() - start2)
    return (qf1inf - qf1inf_val, qf2inf - qf2inf_val, qw1inf - qw1inf_val, qw2inf - qw2inf_val)
equations.counter=0

def epidemic_size():
    qf1inf, qf2inf, qw1inf, qw2inf = fsolve(equations, (0.5, 0.5, 0.5, 0.5), xtol=1e-3)#xtol=1e-10
    #print(time.time())
    print('optimal qs', qf1inf, qf2inf, qw1inf, qw2inf)
    Q, Q1_val, Q2_val = obtain_val_Q(qf1inf, qf2inf, qw1inf, qw2inf)
    #print(time.time() - start)
    #print(Q, Q1_val, Q2_val)
    return Q1_val, Q2_val


file = open('./logfile_epidemic_size_{0}_{1}'.format(lambda_f, lambda_w), mode='w+')


print(lambda_f,lambda_w)
start_0 = time.time()
Q1_final, Q2_final = epidemic_size()
results = 'numNodes: {0} lambdaf: {1} lambdaw: {2} Tf1: {3} Tf2: {4} Tw1: {5} Tw2: {6} muf11: {7} muf22: {8} muw11: {9} muw22: {10} Q1: {11} Q2: {12} max_degree:{13}   \n'.format('infinite', lambda_f , lambda_w, tf1, tf2,tw1, tw2, muf11, muf22, muw11, muw22, Q1_final, Q2_final, max_degree)
print(results)
file.write(results)
file.flush()
print("for one data point total time taken",time.time() - start_0)


# '''Note: For different degree distributions modify the pmf as below:
# # When the degree distribution of layer-f is a mixture of Poisson(poisson_mean_f) and 0,
# # for the obtain_val_qfinf function modify the prob_f as: 
# #     prob_f = (1-p_0_f)*poisson.pmf(d_f, poissson_mean_f)
# # for the obtain_val_qwinf, obtain_val_Q function modify the prob_f as: 
# #     prob_f = (1-p_0_f)*poisson.pmf(d_f, poissson_mean_f)
# #     if (d_f==0):
# #         prob_f += p_0_f, 

# # For the power law degree distribution, to speed up computations precompute the pmf values by importing the polylog module from math
# # def powerlaw_expcutoff_pmf(k,alpha, exp_param):
# #     if k==0:
# #         return 0
# #     else:
# #         num =  float(math.exp(-1.0*k / exp_param)*np.power(k, -1*alpha))
# #         denom = float(polylog(alpha, math.exp(-1.0 / exp_param)))
# #         return num/denom
# '''


