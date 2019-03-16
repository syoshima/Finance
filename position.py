# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:52:26 2019
@author: Samy Abud Yoshima
"""
import numpy as np
import pandas as pd
import math
from pretty_table import get_pretty_table
#from math import exp, log, sqrt, pi
#from statistics import norm_pdf, norm_cdf
from matplotlib import pyplot as plt
from BS_div import div_price_BS, div_delta_BS,div_gamma_BS,div_vega_BS,div_theta_BS,div_rho_BS
from vol_imp import vIP
import datetime

# Load transactions, assets, exchanges, brokers
data = pd.ExcelFile("vale3.xls") 
print(data.sheet_names)

# Define the columns to be read
transa = ['Name','Date','Time', 'Broker', 'Price','Costs Exchange','Costs Broker','WTT','Settlement Date']
data_pos = data.parse(u'Transactions',  names=transa)
data_pos = data_pos.transpose()
data_pos =  data_pos.values
# Variables and Parameters
name     = data_pos[0]             
date_tr  = data_pos[1]
time_tr  = data_pos[2]
broker   = data_pos[3]
price    = data_pos[4]
costs_x  = data_pos[5]
costs_b  = data_pos[6]
wtt      = data_pos[7]
date_set = data_pos[8]
#data_opt.set_index('strike', inplace=True)

# Define the columns to be read
mkt = ['Name','Bid','Offer','Meio','Time Stamp']
data_mkt = data.parse(u'MktP',  names=mkt)
data_mkt = data_mkt.transpose()
data_mkt =  data_mkt.values
# Variables and Parameters
name_mkt  = data_mkt[0]             
bid       = data_mkt[1]
offer     = data_mkt[2]
meio      = data_mkt[3]
timestamp = data_mkt[4]
 
# definir dia de hj e ver posições por dia
# today = input('today')
port = list()
for asset in name:
    if asset not in port:
        port.append(asset)
port.sort()
print('Assets in the portfolio: ',port)


for at in port:
    pos = np.sum([(w) for w, n in zip(wtt, name) if n == at])
      # sum for all names and date_tr >= today and time >= time_tr
    fin = np.sum([w*p - x - b for w,p,x,b,n in zip(wtt,price,costs_x,costs_b,name) if n == at])
    if pos == 0:
        pm = 0
        fin_mkt = 0
        pl = (fin)
    else:
        pm = fin/pos
        fin_mkt = float(np.array([pos*m for m,n in zip(meio,name_mkt) if n == at])) 
        pl = np.array(fin_mkt - fin)
    print(at, pos, fin, pm,fin_mkt,pl)
   # print get_pretty_table([at, pos, fin, pm,fin_mkt,pl], ['Asset', 'Position', 'Total Cost', 'Avg Price','Market Value','P&L'])
# print table
    
# datetime format (escolher dias das posições)

# Positions in BRL USD and CHF
# sum transactions for assets and multiply by market prices

# Risks
# por underlying asset: calculate and sum deltas
# por opt em cada underlying asset , gammas, vegas, thetas e rhos for all assets in the portfolio at a certain date


# P&L
# Calculate market prices minus costs to exit minus price of acwuisition


# Price Predcition and risks 

