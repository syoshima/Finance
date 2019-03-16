# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:06:39 2019

@author: Samy Abud Yoshima
"""
# Dividends Options prices - Closed end formula by Black & Scholes
# Call - Price of a European call option struck at K, withspot S, constant rate r, constant vol v (over the life of the option) and time to maturity T
# PUT - Price of a European put option struck at K, with spot S, constant rate r, constant vol v (over the life of the option) and time to maturity T

import numpy as np
import math
from math import exp, log, sqrt, pi
from statistics import norm_pdf, norm_cdf

def d_j(j, S, K, r, d, v, T):
    """
    d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)} where j∈{1,2} .
    """
    return (log(S/K) + (r - d +((-1)**(j-1))*0.5*v*v)*T)/(v*(T**0.5))


def div_price_BS(S,K,r,d,v,T,p):
    if T <= 0:
        if p == "call":
            return max (0, S - K)
        elif p == "put":
            return max (0, K - S)
    else:
        if p == "call":
            return S * exp(-d*T) * norm_cdf(d_j(1, S, K, r, d, v, T)) -  K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, d, v, T))
        elif p == "put":
            return -S * exp(-d*T)*norm_cdf(-d_j(1, S, K, r, d, v, T)) + K*exp(-r*T) * norm_cdf(-d_j(2, S, K, r, d, v, T))
    
# Delta, Gamma, Vega, Theta, Rho
def div_delta_BS(S,K,r,d,v,T,p):
    if T <= 0:
        if p == "call":
            return max (0, S - K)/max(0.0001,S-K)
        elif p == "put":
            return - max(0, K - S)/max(0.00001,K-S)  
    else:
        if p == "call":
            return norm_cdf(d_j(1, S, K, r, d, v, T))
        elif p == "put":
            return - norm_cdf(-d_j(1, S, K, r, d, v, T))

def div_gamma_BS(S,K, r,d,v,T,p):
    if T <= 0:
        return 0
    else:
        if p == "call":
            return norm_pdf(d_j(1, S, K, r, d, v, T)) / (S*v*T**0.5)
        elif p == "put":
            return norm_pdf(d_j(1, S, K, r, d, v, T)) / (S*v*T**0.5)

def div_vega_BS(S,K,r, d,v,T,p):
    if T <= 0:
        return 0
    else:
        if p == "call":
            return S * T**0.5 * norm_pdf(d_j(1,S,K,r,d,v,T))
        elif p == "put":
            return S * T**0.5 * norm_pdf(d_j(1,S,K,r,d,v,T))
        
def div_theta_BS(S,K,r,d,v,T,p):
    if T <= 0:
        return 0  
    else:
        if p == "call":
            return (S*v*norm_pdf(d_j(1,S,K,r,d,v,T)))/(2*T**0.5) - r*K*exp( -r*T)*norm_cdf(d_j(2,S,K,r,v,T))
        elif p == "put":
            return -(S*v*norm_pdf(d_j(1,S,K,r,d,v,T)))/(2*T**0.5) + r*K*exp( -r*T)*norm_cdf(-d_j(2,S,K,r,d,v,T))
            
def div_rho_BS(S, K, r, d, v, T,p):
    if T <= 0:
        return 0    
    else:
        if p =="call":
            return K*T*exp(-r*T)*norm_cdf(d_j(2,S,K,r,d,v,T))
        elif p=="put":
            return - K*T*exp(-r*T)*norm_cdf(-d_j(2,S,K,r,d,v,T))