# Option Pricing
import numpy as np
import math
#from math import exp, log, sqrt, pi
#from statistics import norm_pdf, norm_cdf
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from blackscholes_dividends import dividends_call_price_BS, dividends_put_price_BS,dividends_call_delta_BS,dividends_call_gamma_BS,dividends_call_vega_BS,dividends_put_delta_BS, dividends_put_gamma_BS, dividends_put_vega_BS
#import datetime

#from business_calendar import Calendar, MO, TU, WE, TH, FR
#date1 = datetime.datetime.today()
#date2 = datetime.datetime(2019,2,8)
#cal = Calendar(workdays=[MO,TU,WE,TH,FR])
#cal.busdaycount(date1, date2), date1, date2)
#cal.addbusdays(date1, 25)

# Variables
S = 56 * (1 - 0.08)*(1-0.05)                #: Spot Price
K = 47.14                #: Strike Price
r =  0.065 / (252)       #: Interest rate daily
# Para estimar volatilidade diaria, usar todos os negócios. Criar medida para ponderar por wtt negociada. Comparar tbm com max e min.
d = 0.1 / (252) # dividends payout
v =  0.4 / (252 ** 0.5) #: Daily Volatility
T = 15                   #: cal.busdayscount(date1, date2) #: time to maturity in business days
call_mkt_price = 7.0     #:
put_mkt_price  =  0.18   #:

    


# Implied Vol
vIC = v
difC = call_mkt_price - dividends_call_price_BS(S, K, r, d, vIC, T)
while abs(difC) > 0.0001 :
    difC = call_mkt_price - dividends_call_price_BS(S, K, r, d, vIC, T)
    if difC > 0:
       vIC = vIC + max(1,(abs(difC*1000)/abs(call_mkt_price)))*0.0000001
    elif difC < 0:
        vIC = vIC - max(1,(abs(difC*1000)/abs(call_mkt_price)))*0.0000001
print("Call Price = ","{:10.2f}".format( float(dividends_call_price_BS(S, K, r, d, v, T))))
print("Call Price = ",call_mkt_price,"Implied Vol.", "{:10.2f}".format(float((vIC*100)*252**0.5)),"%")
# Analysis
call_int_value = max(0,S - K)
call_time_value = call_mkt_price - call_int_value
print("Intrinsic Value =","{:10.2f}".format(call_int_value)," / Time Vale =","{:10.4f}".format(call_time_value))
print("Theta ´per day in BRL =","{:10.2f}".format(call_time_value/T))
# Vega, Delta
# Graph pay-off at maturity

#Axis
varp = 0.2 # define axis' values
prices = np.arange(int(S*(1-varp)),int(S*(1+varp)),1)
dates = np.arange(0,T,1)
ops = [[round(float(dividends_call_price_BS(price, K, r, d, vIC, days)),2) for price in prices] for days in dates]
min_ops = max(0,round(min((min(ops))),0)-1)
max_ops =  round(max((max(ops))),0)- 1
ops = np.array(ops)
tvds = [[round(float(dividends_call_price_BS(price, K, r, d, vIC, days)-max(0,price-K)),2) for price in prices] for days in dates]
tvds= np.array(tvds)

# Graph superficie option price, spot e date
fig = plt.figure(1)
X = prices
Y = dates
Z = ops
min_Z = min_ops
max_Z = max_ops
ax = fig.add_subplot(2, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
surf= ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)   
# Customize the z axis.
ax.set_zlim(min_Z,max_Z)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Graph superficie time value per day, spot e date
X = prices
Y = dates
Z = tvds
ax = fig.add_subplot(2, 1, 2, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=2)


# Graph superficie call price, spot e time value


# PUT

# Implied Vol
vIP = v
difP = put_mkt_price - dividends_put_price_BS(S, K, r, d, vIP, T)
while abs(difP) > 0.0001 :
    difP = put_mkt_price - dividends_put_price_BS(S, K, r, d, vIP, T)
    if difP > 0:
       vIP = vIP + max(1,(abs(difP*1000)/abs(put_mkt_price)))*0.0000001
    elif difP < 0:
       vIP = vIP - max(1,(abs(difP*1000)/abs(put_mkt_price)))*0.0000001
print("Put Price = ", "{:10.2f}".format(float(dividends_put_price_BS(S, K, r, d, v, T))))
print("Put MktPr = ",put_mkt_price,"Implied Vol.", "{:10.4f}".format(float(vIP*100)*252**0.5),"%")
# Analysis
put_int_value = max(0,K - S)
put_time_value = put_mkt_price - put_int_value
print('Spot',S, "Strike Put", K)
print("Intrinsic Value =","{:10.2f}".format(put_int_value)," / Time Vale =","{:10.4f}".format(put_time_value))
print("Theta ´per day in BRL =","{:10.2f}".format(put_time_value/T))
print("Delta:",round(float(dividends_put_delta_BS(S, K, r, d, vIC, T)),2),"Gamma:",round(float(dividends_put_gamma_BS(S, K, r, d, vIC, T)),2))
print("Vega:",round(float(dividends_put_vega_BS(S, K, r, d, vIC, T)),2)) 


#Axis for PUT
varp = 0.2 # define axis' values
prices = np.arange(int(S*(1-varp)),int(S*(1+varp)),1)
dates = np.arange(0,T,1)
vols = np.arange((vIP*(1-varp)),(vIP*(1+varp)),0.001)
ops = [[round(float(dividends_put_price_BS(price, K, r, d, vIC, days)),2) for price in prices] for days in dates]
min_ops = max(0,round(min((min(ops))),0)-1)
max_ops =  round(max((max(ops))),0)- 1
ops = np.array(ops)
tvds = [[round(float(dividends_put_price_BS(price, K, r, d, vIC, days)-max(0,K-price)),2) for price in prices] for days in dates]
tvds= np.array(tvds)
deltas = [[round(float(dividends_put_delta_BS(price, K, r, d, vIC, days)),2) for price in prices] for days in dates]
deltas = np.array(deltas)
gammas = [[round(float(dividends_put_gamma_BS(price, K, r, d, vol, T)),2) for price in prices] for vol in vols]
gammas = np.array(gammas)
vegas = [[round(float(dividends_put_vega_BS(price, K, r, d, vol, T)),2) for price in prices] for vol in vols]
vegas = np.array(vegas)

# Graph superficie option price, spot e date
fig = plt.figure(2)
# Define
X = prices
Y = dates
Z = ops
min_Z = min_ops
max_Z = max_ops
# Plot
ax = fig.add_subplot(2, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)   
# Customize the z axis.
ax.set(xlabel='price', ylabel='time',zlabel='Put_prices',title='PUT price')
ax.set_zlim(min_Z,max_Z)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Graph superficie time value per day, spot e vols
X = prices
Y = vols* 252 ** 0.5
Z = vegas
ax = fig.add_subplot(2, 1, 2, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=2)
ax.set(xlabel='price', ylabel='vols',zlabel='vegas',title='Vgeas´PUT today')

# Graph superficie option delta, spot e dates
fig = plt.figure(3)
X = prices
Y = dates
Z = deltas
ax = fig.add_subplot(2, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)   
ax.set(xlabel='price', ylabel='time',zlabel='delta',title='Deltas´PUT today')
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Graph superficie gammas, spot e date
X = prices
Y = vols* 252 ** 0.5
Z = gammas
ax = fig.add_subplot(2, 1, 2, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=2)
ax.set(xlabel='price', ylabel='vols',zlabel='gamma',title='Gammas´PUT today')

# Graph pay-off at maturity
fig = plt.figure(4)
X = prices
Z = [round(float(dividends_put_price_BS(price, K, r, d, vIC, T)),2) - put_mkt_price for price in prices]
ax = fig.add_subplot(1,1,1)
ax.plot(X,Z,  color='green', marker='o', linestyle='dashed',linewidth=2, markersize=2)
ax.set(xlabel='price', ylabel='payoff',title='At maturity')
ax.grid()
fig.savefig("test.png")
plt.show()
