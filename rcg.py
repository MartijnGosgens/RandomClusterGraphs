import numpy as np
import math
from mpmath import mp
import matplotlib.pyplot as plt

# This computes the n-th Bell number using Dobinski's formula. It seems to be a bit imprecise for n>22, 
# which appears to be because of the finite precision of Python's math.e 
def bell(n): 
    k = 0
    value = 0
    while True:
        k += 1
        fact_log = mp.log(math.factorial(k))
        value += mp.exp(mp.log(k)*n-fact_log)
        if mp.log(k)*n<fact_log:
            return mp.ceil(value / math.e)

# We define B_n(w) as the sum ofw^{#edges} summed over all possible 
# cluster graphs of n vertices. Hence, B_n(1) is the Bell number and we have the recursion 
# B_n(w)=\sum_{s=1}^n w^{s\choose 2}{n-1\choose n-s}B_{n-s}(w). This function returns the list
# [B_0(w),B_1(w),...,B_n(w)]
def bell_generalized_list(n,p=1/2):
    bells = {0:1}
    w=p/(1-p)
    alpha = mp.log(p)-mp.log(1-p)
    for i in range(1,n+1):
        bells[i] = sum([mp.exp(alpha*math.comb(s,2))*math.comb(i-1,s-1)*bells[i-s] for s in range(1,i+1)])
    return bells

# Samples from the size-biased clique distribution for n vertices and ER-probability p. The probability that
# the clique of a uniformly chosen vertex has size s is given by
# (p/(1-p))^{s\choose 2}{n-1\choose n-s} B_{n-s}(p/(1-p))/B_n(p/(1-p)).
def random_s(n,p=1/2,bells=None):
    # We use the recursive implementation for p!=1/2 and Dobinski's formula for p=1/2
    B = bell
    if bells is not None:
        B = lambda n: bells[n]
    elif p != 1/2:
        bells = bell_generalized_list(n,p=p)
        B = lambda n: bells[n]
    thres = np.random.rand()*B(n)
    s = 0
    while s<n:
        s += 1        
        subtr = math.comb(n-1,s-1) * B(n-s) # {n-1 \choose n-s} * B_{n-s}(p/(1-p))
        if p!=1/2:
            subtr*=mp.exp(mp.log(p/(1-p))*math.comb(s,2)) # (p/(1-p))^{s \choose 2}
        thres -= subtr
        if thres <0:
            return s
    print(thres,'leftover')
    return n

# Returns a sample of clique sizes for the ER(n,p) graph conditioned on each component being a fully connected
def random_sizes(n,p=1/2,bells=None):
    if n==0:
        return []
    s = random_s(n,p=p,bells=bells)
    return [s]+random_sizes(n-s,p=p,bells=bells)

def sizes2mG(sizes):
    return sum(s*(s-1)/2 for s in sizes)

def t2p(t):
    return 1/(1+mp.exp(-t))

def p2t(p):
    return mp.log(p/(1-p))

# Recursively computes the expected number of edges.
def means_list(n,p,bells=None):
    if bells is None:
        bells = bell_generalized_list(n,p=p)
    means = {0:0}
    w=p/(1-p)
    alpha = mp.log(p)-mp.log(1-p)
    for i in range(1,n+1):
        means[i] = sum([
            mp.exp(alpha*math.comb(s,2)) * math.comb(i-1,s-1) * bells[i-s] * (math.comb(s,2) + means[i-s])
            for s in range(1,i+1)
        ]) / bells[i]
    return means

# Recursively computes the variance of the number of edges.
def variances_list(n,p,bells=None,means=None):
    if bells is None:
        bells = bell_generalized_list(n,p=p)
    if means is None:
        means = means_list(n,p=p,bells=bells)
    variances = {0:0}
    w=p/(1-p)
    alpha = mp.log(p)-mp.log(1-p)
    for i in range(1,n+1):
        variances[i] = sum([
            mp.exp(alpha*math.comb(s,2)) * math.comb(i-1,s-1) * bells[i-s] * ((math.comb(s,2) + means[i-s])**2+variances[i-s])
            for s in range(1,i+1)
        ]) / bells[i] - means[i]**2
    return variances

# Computes the probability of obtaining the complete graph.
def probcomplete(n,p=1/2,bells=None):
    if bells is None:
        bells = bell_generalized_list(n,p=p)
    t = mp.log(p/(1-p))
    return mp.exp(t*math.comb(n,2))/bells[n]
    
# Computes the expected number of edges assuming p=1/2.
def mean_p_half(n,bells=None):
    if bells is None:
        bells = bell_generalized_list(n)
    return math.comb(n,2)*bells[n-1]/bells[n]

# Computes the variance of the number of edges assuming p=1/2.
def variance_p_half(n,bells=None):
    if bells is None:
        bells = bell_generalized_list(n)
    N=math.comb(n,2)
    return N**2 * (bells[n-1]/bells[n]) * (bells[n-2]/bells[n-1] - bells[n-1]/bells[n])+ N * (bells[n-1]/bells[n]) * (1-bells[n-1]/bells[n])
    
# We use Newton-Raphson iteration to find the value of t for which B_n(e^t)==target.
def B_inverse_nr(n,target,tolerance=1/1000):
    t=mp.log(target)/math.comb(n,2)
    p=mp.exp(t)/(1+mp.exp(t))
    bells = bell_generalized_list(n,p)
    val = mp.log(bells[n])-mp.log(target)
    while abs(val)>tolerance:
        means = means_list(n,p)
        t = t - val/means[n]
        p=mp.exp(t)/(1+mp.exp(t))
        bells = bell_generalized_list(n,p)
        val = mp.log(bells[n])-mp.log(target)
    return t

# We use Newton-Raphson iteration to find the value of t for which probcomplete(n,t2p(t))==target.
def probcomplete_inverse_nr(n,target=1/2,tolerance=1/1000,silent=True):
    if not silent:
        print('Inverting probcomplete={} for n={}'.format(target,n))
    N=math.comb(n,2)
    t=mp.log(target*bell(n))/N
    p=mp.exp(t)/(1+mp.exp(t))
    bells = bell_generalized_list(n,p)
    val = N*t-mp.log(bells[n])-mp.log(target)
    while abs(val)>tolerance:
        means = means_list(n,p)
        t = t - val/(N-means[n])
        p=mp.exp(t)/(1+mp.exp(t))
        bells = bell_generalized_list(n,p)
        val = N*t-mp.log(bells[n])-mp.log(target)
        if not silent:
            print('probcomplete',t,val)
    return t

# We use Newton-Raphson iteration to find the value of t for which n*B_{n-1}(e^t)/B_n(e^t)==target.
def kappa_inverse_nr(n,target,tolerance=1/1000,silent=True):
    if not silent:
        print('Inverting kappa={} for n={}'.format(target,n))
    N=math.comb(n,2)
    t = 0 if target>=mp.lambertw(n) else probcomplete_inverse_nr(n,silent=silent)
    p=t2p(t)
    bells = bell_generalized_list(n,p)
    val = mp.log(n)+mp.log(bells[n-1])-mp.log(bells[n])-mp.log(target)
    while abs(val)>tolerance:
        means = means_list(n,p)
        t = t - val/(means[n-1]-means[n])
        p=t2p(t)
        bells = bell_generalized_list(n,p)
        val = mp.log(n)+mp.log(bells[n-1])-mp.log(bells[n])-mp.log(target)
        if not silent:
            print('kappa',t,val)
    return t

# We use Newton-Raphson iteration to find the value of t for which the expected number of edges is equal to target.
def m_inverse_nr(n,target,tolerance=1/1000,silent=True):
    # Since we are inverting a monotone function, we can keep track of the upper and lower bounds
    # This is very helpful to improve the convergence, as NR has the tendency to overshoot and
    # diverge away from the solution when the derivative (the Var(mG)) isn't well-behaved (and it isn't)
    # We get some kind of hybrid between NR and the bisection method.
    upper = float('inf')
    lower = -float('inf')
    t=0
    N=math.comb(n,2)
    # Asymptotically equivalent to E[mG]
    thresh = n*(mp.log(n)-mp.log(mp.log(n)))/2
    if target>thresh:
        t=probcomplete_inverse_nr(n,target/N,silent=silent)
        if not silent:
            print('initial guess: ',t)
    if target<thresh:
        s = 2*target/n+2 # 'typical' cluster size
        t = -2*mp.log(n)/(s**2-s)
        if n>210:
            upper = 0 # For n>210, bell(n) already gives OverflowError
    p=mp.exp(t)/(1+mp.exp(t))
    bells = bell_generalized_list(n,p)
    means = means_list(n,p,bells=bells)
    val = means[n]-target
    while abs(val)>tolerance:
        if val>0:
            upper = min(upper,t)
        if val<0:
            lower = max(lower,t)
        variances = variances_list(n,p,bells=bells,means=means)
        change = val/variances[n]
        if t-change>upper:
            if not silent:
                print('NR wanted',t-change,'but reduced to',(t+upper)/2)
            t=(t+upper)/2
        elif t-change<lower:
            if not silent:
                print('NR wanted',t-change,'but increased to',(t+lower)/2)
            t=(t+lower)/2
        else:
            t = t - change
        p=mp.exp(t)/(1+mp.exp(t))
        bells = bell_generalized_list(n,p)
        means = means_list(n,p,bells=bells)
        val = means[n]-target
        if not silent:
            print('m',t,val)
    return t

# Returns the resolution parameter for which modularity maximization is equivalent to Bayesian inference of a PPM.
def bayesian_gamma(p_in,p_out,p):
    return mp.log((1-p_out)/(1-p_in) * (1-p)/p)/mp.log(p_in / p_out * (1-p_out) / (1-p_in))