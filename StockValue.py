import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def monteCarlo(S,r,T,q,sigma,steps,N):
    # S     = stock price
    # r     = risk free rate
    # T     = time left til maturity (years)
    # q     = dividend yield
    # sigma = volatility
    # steps = time steps
    # N     = number of trials
    
    dt = T/steps
    ST = np.log(S) + np.cumsum(((r - q - (sigma**2)/2) * dt + sigma*np.sqrt(dt) * np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)
    

def blackScholes(S,K,r,T,q,sigma):
    
    # S     = stock price
    # K     = strike price
    # r     = risk free rate
    # T     = time left til maturity (years)
    # q     = dividend yield
    # sigma = volatility
    
    d1 = (np.log(S/K) + (r - q + (sigma**2)/2 ) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(d2)
    
    return call



S     = 100
K     = 110
T     = 1/2
r     = 0.05
q     = 0.02
sigma = 0.25
steps = 100
N     = 1000 # as this number approaches infinity the closer it'll be

path = monteCarlo(S,r,T,q,sigma,steps,N)

plt.plot(path)
plt.xlabel("Time Increments")
plt.ylabel("Stock Price")
plt.title("Geometric Brownian Motion")

payoffs = np.maximum(path[-1]-K, 0)
option_price = np.mean(payoffs)*np.exp(-r*T)

scholesPrice = blackScholes(S,K,r,T,q,sigma)

print("Black Scholes price is ", scholesPrice)
print("Simulated Price is ", option_price)