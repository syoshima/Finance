# Option Pricing
import numpy as np
import math
from math import exp, log, sqrt, pi
from statistics import norm_pdf, norm_cdf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import datetime

#from business_calendar import Calendar, MO, TU, WE, TH, FR
#date1 = datetime.datetime.today()
#date2 = datetime.datetime(2019,2,8)
#cal = Calendar(workdays=[MO,TU,WE,TH,FR])
#cal.busdaycount(date1, date2), date1, date2)
#cal.addbusdays(date1, 25)

# Variables
S = 54.0                #: Spot Price
K = 49.                #: Strike Price
r =  0.065 / (252)       #: Interest rate daily
# Para estimar volatilidade diaria, usar todos os negócios. Criar medida para ponderar por wtt negociada. Comparar tbm com max e min.
v =  0.18 / (252 ** 0.5) #: Daily Volatility
T = 20                   #: cal.busdayscount(date1, date2) #: time to maturity in business days
call_mkt_price = 6.0     #:
put_mkt_price  =  0.35   #:

    
def d_j(j, S, K, r, v, T):
    """
    d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)} where j∈{1,2} .
    """
    return (log(S/K) + (r + ((-1)**(j-1))*0.5*v*v)*T)/(v*(T**0.5))

# Call
# BS Price
def vanilla_call_price_BS(S, K, r, v, T):
    """
    Price of a European call option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
    """
    if T <= 0:
        return max (0, S - K)
    else:
        return S * norm_cdf(d_j(1, S, K, r, v, T)) -  K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, v, T))

# Implied Vol
vIC = v
difC = call_mkt_price - vanilla_call_price_BS(S, K, r, vIC, T)
while abs(difC) > 0.0001 :
    difC = call_mkt_price - vanilla_call_price_BS(S, K, r, vIC, T)
    if difC > 0:
       vIC = vIC + max(1,(abs(difC*1000)/abs(call_mkt_price)))*0.0000001
    elif difC < 0:
        vIC = vIC - max(1,(abs(difC*1000)/abs(call_mkt_price)))*0.0000001
print("Call Price = ","{:10.2f}".format( float(vanilla_call_price_BS(S, K, r, v, T))))
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
cps = [[round(float(vanilla_call_price_BS(price, K, r, vIC, days)),2) for price in prices] for days in dates]
min_Z = max(0,round(min((min(cps))),0)-1)
max_Z =  round(max((max(cps))),0)- 1
cps = np.array(cps)
tvds = [[round(float(vanilla_call_price_BS(price, K, r, vIC, days)-max(0,price-K)),2) for price in prices] for days in dates]
tvds= np.array(tvds)

# Graph superficie option price, spot e date
fig = plt.figure()
X = prices
Y = dates
Z = cps
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
# BS Price
def vanilla_put_price_BS(S, K, r, v, T):
    """
    Price of a European put option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
      """
    if T <= 0:
        return max (0, K - S)
    else:
        return -S * norm_cdf(-d_j(1, S, K, r, v, T)) + K*exp(-r*T) * norm_cdf(-d_j(2, S, K, r, v, T))
# Implied Vol
vIP = v
difP = put_mkt_price - vanilla_put_price_BS(S, K, r, vIP, T)
while abs(difP) > 0.0001 :
    difP = put_mkt_price - vanilla_put_price_BS(S, K, r, vIP, T)
    if difP > 0:
       vIP = vIP + max(1,(abs(difP*1000)/abs(put_mkt_price)))*0.0000001
    elif difP < 0:
       vIP = vIP - max(1,(abs(difP*1000)/abs(put_mkt_price)))*0.0000001
print("Put Price = ", "{:10.2f}".format(float(vanilla_put_price_BS(S, K, r, v, T))))
print("Put MktPr = ",put_mkt_price,"Implied Vol.", "{:10.4f}".format(float(vIP*100)*252**0.5),"%")

plt.show()
