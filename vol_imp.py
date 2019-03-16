# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:06:39 2019

@author: Samy Abud Yoshima
"""

import numpy as np
import math
from math import exp, log, sqrt, pi
from statistics import norm_pdf, norm_cdf
from BS_div import div_price_BS, div_price_BS

def vIP(price,S,strike,r,d,v,T,p):
    vIP = v
    k = 0.0001
    difP = price - div_price_BS(S,strike,r,d,vIP,T,p)
    while abs(difP) > k :
        difP = price - div_price_BS(S,strike,r,d,vIP,T,p)
        if difP > 0:
            vIP = vIP + max(1,(abs(difP/k)/abs(price)))*k**2
        elif difP < 0:
            vIP = vIP - max(1,(abs(difP/k)/abs(price)))*k**2
    return vIP