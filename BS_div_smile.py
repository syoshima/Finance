# Option Pricing
import math
import scipy as sp
import pandas as pd
import numpy as np
from statistics import norm_pdf, norm_cdf
from matplotlib import pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from BS_div import div_price_BS, div_delta_BS,div_gamma_BS,div_vega_BS,div_theta_BS,div_rho_BS
from vol_imp import vIP
#import datetime

# load data
data = pd.ExcelFile("vale3.xls") 
#print(data.sheet_names)
# Define the columns to be read
columns1 = ['strike','bid', 'offer', 'halfs','Price Underlying','r','d','v','T','Maturity','Code','Put/Call','Underlying Asset','Type Asset','Exchange','	Country']
data_opt = data.parse(u'1.1',  names=columns1)
data_opt = data_opt.transpose()
data_opt =  data_opt.values
# Variables and Parameters
S = data_opt[4,0]              #S = Spot Price
r = data_opt[5,0]/(252)        #r: Interest rate daily
# Para estimar volatilidade diaria, usar todos os negócios. Criar medida para ponderar por wtt negociada. Comparar tbm com max e min.
d = data_opt[6,0]/(252)        #d: dividends payout
v = data_opt[7,0]/(252**0.5)   #v: Daily Volatility
T = data_opt[8,0]              #T: cal.busdayscount(date1, date2) #: time to maturity in business days
pc = data_opt[11]
#data_opt.set_index('strike', inplace=True)
#print(data_opt)
# Graph
fig = plt.figure(1)
X  = data_opt[0]
Y1 = data_opt[1]
Y2 = data_opt[2]
Y3 = data_opt[3]
ax = fig.add_subplot(1,1,1)
plt.plot(X, Y1,  marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=2,label="Bid")
plt.plot(X, Y2,  marker='', color='olive', linewidth=2,label="Offer")
plt.plot(X, Y3,  marker='', color='olive', linewidth=2, linestyle='dashed', label="Meio")
plt.title('Put premia prices')
plt.legend()
ax.grid()
fig.savefig("strikes.png")
# Smile
strikes = data_opt[0]
bids = data_opt[1]
offers = data_opt[2]
halfs = data_opt[3]
smile_put_bids   = [float(vIP(bid,S,strike,r,d,v,T,p))   for bid,strike,p   in zip(bids,strikes,pc) if p == "put"] 
smile_put_offers = [float(vIP(offer,S,strike,r,d,v,T,p)) for offer,strike,p in zip(offers,strikes,pc) if p == "put"] 
smile_put_halfs  = [float(vIP(half,S,strike,r,d,v,T,p))  for half,strike,p  in zip(halfs,strikes,pc) if p == "put"] 
smile_put_bids_a   = [round(x*(252**0.5)*100,2) for x in smile_put_bids]
smile_put_offers_a = [round(x*(252**0.5)*100,2) for x in smile_put_offers]
smile_put_halfs_a  = [round(x*(252**0.5)*100,2) for x in smile_put_halfs]

# Graph
fig = plt.figure(2)
x  = strikes
y1 = smile_put_bids_a
y2 = smile_put_offers_a
y3 = smile_put_halfs_a
plt.plot(x,y1,marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=2,label="Bid")
plt.plot(x,y2,marker='', color='olive', linewidth=2,label="Offer")
plt.plot(x,y3,marker='', color='olive', linewidth=2, linestyle='dashed', label="Meio")
plt.title('Put smiles')
plt.legend(loc='best')
ax.grid()
fig.savefig("smiles.png")

#Interpolation Linear and Cubic of Implied Vol Smile
fig = plt.figure(3)
x = np.array(strikes)
y = np.array(smile_put_halfs)
f = interpolate.interp1d(x, y)
xnew = np.linspace((min(x)),(max(x)),num=(max(x)-min(x))*100, endpoint=True)
tck = interpolate.splrep(x, y, s=0)
ynew = interpolate.splev(xnew, tck, der=0)
y = [round(float(x)*(252**0.5)*100,22) for x in y]
fxnew = [round(float(x)*(252**0.5)*100,22) for x in f(xnew)]
ynew = [round(float(x)*(252**0.5)*100,22) for x in ynew]
y2 = np.array(ynew)-np.array(fxnew)
color = 'tab:red'
ax1 = np.array(xnew)
fig, ax1 = plt.subplots()
ax1.set_xlabel('strike')
ax1.set_ylabel('volimp', color=color)
ax1.plot(x, y, 'o', xnew, fxnew, '-', xnew, ynew, 'b')
plt.legend(['data', 'linear', 'cubic'], loc='best')
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('cubic -linear', color=color)  # we already handled the x-label with ax1
ax2.plot(xnew,y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clippedplt.legend(['Data','Linear', 'Cubic Spline'])
plt.legend(['dif'], loc='best')
plt.title('Put Bid-Offer Avg Smile plus Linear and Cubic-spline interpolation')
# interpolation 2 Aprox
fig = plt.figure(4)
fig, ax1 = plt.subplots()
x = np.array(strikes)
y = np.array(smile_put_halfs)
xnew = np.linspace((min(x)),(max(x)),num=(max(x)-min(x))*100, endpoint=True)
f1 = interpolate.interp1d(x, y, kind='nearest') 
f2 = interpolate.interp1d(x, y, kind='zero') 
f3 = interpolate.interp1d(x, y, kind='slinear')
xnew = np.linspace((min(x)),(max(x)),num=100, endpoint=True)
f1 = [round(x*(252**0.5)*100,22) for x in f1(xnew)]
f2 = [round(x*(252**0.5)*100,22) for x in f2(xnew)]
f3 = [round(x*(252**0.5)*100,22) for x in f3(xnew)]
y = smile_put_halfs_a
plt.plot(x, y, 'o', xnew, f1, '-', xnew, f2, '--', xnew, f3, ':')
plt.legend(['data', 'nearest', 'zero', 'slinear'], loc='best')
plt.title('Put Bid-Offer Avg Smile Aprox interpolation')

## Graph BS Price for puts
fig = plt.figure(5)
fig, ax1 = plt.subplots()
p = 'put'
vf = 0.63/(252**0.5)
x = np.array(strikes)
y = np.array(smile_put_halfs)
xnew = np.linspace((min(x)),(max(x)),num=(max(x)-min(x))*100, endpoint=True)
xnew = [round(x,2) for x in xnew]
tck = interpolate.splrep(x, y, s=0)
ynew = interpolate.splev(xnew, tck, der=0)
put_prices_smile = [round(float(div_price_BS(S,K,r,d,v,T,p)),2) for K,v in zip(xnew,ynew) ]
put_prices_vf = [round(float(div_price_BS(S,K,r,d,vf,T,p)),2) for K in xnew ]
y = halfs
f4 = np.array(put_prices_smile)
f5 = np.array(put_prices_vf)
plt.plot(x, y, 'o', xnew, f4, '--', xnew, f5, ':')
plt.legend(['Mkt halfs', 'BS Put Prices','Fixed Vol'], loc='best')
plt.title('Put Prices from Cubic Spline Implied Vol Smile interpolation')
plt.show()
## Choose a strike to get price
k = 43.14
put_price = [round(float(div_price_BS(S,K,r,d,v,T,p)),2) for K,v in zip(xnew,ynew) if K == k]
put_price = np.array(put_price)
put_price_vol = np.array(round(float(vIP(put_price,S,k,r,d,v,T,p))*252**0.5*100,2))
## Print stuff
print("Strike",k,"Put Price",put_price,"Vol",put_price_vol,"%")
print(strikes)
print(Y1)
print(smile_put_bids_a)
print(Y2)
print(smile_put_offers_a)
print(Y3)
print(smile_put_halfs_a)

#Axis for PUT
K = k
varp = 0.5 # define axis' values
vIP =  np.array(round(float(vIP(put_price,S,k,r,d,v,T,p)),2))
prices = np.arange(int(S*(1-varp)-1),int(S*(1+varp)+1),0.1)
prices = [round(float(x),2) for x in prices]
dates = np.arange(0,T,1)
vols = np.arange(round(vIP*(1-varp),2),round(vIP*(1+varp),2),0.01)
ops = [[round(float(div_price_BS(price, K, r, d, vIP, days,p)),2) for price in prices] for days in dates]
min_ops = max(0,round(min((min(ops))),0)-1)
max_ops =  round(max((max(ops))),0)- 1
ops = np.array(ops)
tvds = [[round(float(div_price_BS(price,K,r,d,vIP,days,p)-max(0,K-price)),2) for price in prices] for days in dates]
tvds= np.array(tvds)
deltas = [[round(float(div_delta_BS(price,K,r,d,vIP,days,p)),2) for price in prices] for days in dates]
deltas = np.array(deltas)
gammas = [[round(float(div_gamma_BS(price,K,r,d,vol,T,p)),2) for price in prices] for vol in vols]
gammas = np.array(gammas)
vegas = [[round(float(div_vega_BS(price,K,r,d,vol,T,p)),2) for price in prices] for vol in vols]
vegas = np.array(vegas)
# Graph superficie option price, spot e date
fig = plt.figure(6)
X = prices
Y = dates
Z = ops
min_Z = min_ops
max_Z = max_ops
# Plot
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)   
# Customize the z axis.
ax.set(xlabel='price', ylabel='time',zlabel='Prices',title='PUT price')
ax.set_zlim(min_Z,max_Z)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Graph superficie time value per day, spot e vols
fig = plt.figure(7)
X = prices
Y = vols * 252 ** 0.5 * 100
Z = vegas
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=2)
ax.set(xlabel='price', ylabel='vols',zlabel='vegas',title='Vgeas´PUT today')

# Graph superficie option delta, spot e dates
fig = plt.figure(8)
X = prices
Y = dates
Z = deltas
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)   
ax.set(xlabel='price', ylabel='time',zlabel='delta',title='Deltas´PUT today')
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Graph superficie gammas, spot e date
fig = plt.figure(9)
X = prices
Y = vols * 252 ** 0.5 * 100
Z = gammas
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=2)
ax.set(xlabel='price', ylabel='vols',zlabel='gamma',title='Gammas´PUT today')

# Graph pay-off at maturity
fig = plt.figure(10)
X = prices
Z = [round(float(div_price_BS(price, K, r, d, vIP, T,p)),2) - put_price for price in prices]
ax = fig.add_subplot(1,1,1)
ax.plot(X,Z, color='green', marker='o', linestyle='dashed',linewidth=2, markersize=2)
ax.set(xlabel='price', ylabel='payoff',title='At maturity')
ax.grid()
fig.savefig("test.png")
plt.show()
# Analysis
put_price = np.array(put_price)
put_int_value = np.array(round(max(0,K - S),2))
put_time_value = np.array(round(float(put_price) - float(put_int_value),2))
print("Put Price = ",round(float(div_price_BS(S, K, r, d, vIP, T,p)),2),"vol", round(vIP*252**0.5*100,2),"%")
print("Put MktPr = ",round(float(put_price),2),"Implied Vol.", round(float(vIP)*100*252**0.5,2),"%")
print('Spot',S, "Strike Put", K)
print("Intrinsic Value =",round(float(put_int_value),2),"Time value= ",round(float(put_time_value),2))
print("Theta ´per day in BRL =",round(float(put_time_value/T),2))
print("Delta:",round(float(div_delta_BS(S, K, r, d, vIP, T,p)),2),"Gamma:",round(float(div_gamma_BS(S, K, r, d, vIP, T,p)),2),"Vega:",round(float(div_vega_BS(S, K, r, d, vIP, T,p)),2)) 

# use smile to price other options, interpolate smile
# 0) Import libs
# 1) Load data
# 2) input parameters
# 3) Calculate smile, interpoelate, calculate prices of puts

# Analysis
#put_int_value = max(0,K - S)
#put_time_value = put_mkt_price - put_int_value
#print('Spot',S, "Strike Put", K)
#print("Intrinsic Value =","{:10.2f}".format(put_int_value)," / Time Vale =","{:10.4f}".format(put_time_value))
#print("Theta ´per day in BRL =","{:10.2f}".format(put_time_value/T))
#print("Delta:",round(float(dividends_put_delta_BS(S, K, r, d, vIP, T)),2),"Gamma:",round(float(dividends_put_gamma_BS(S,K,r,d,vIP,T)),2))
#print("Vega:",round(float(dividends_put_vega_BS(S, K, r, d, vIP, T)),2)) 
#PUTs= [round(dividends_put_price_BS(S, strike, r, d, vIP, T),2) for strike in strikes]
#print(strikes)
#print(PUTs)