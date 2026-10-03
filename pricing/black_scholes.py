import numpy as np
from scipy.stats import norm

def black_scholes(S0, K, T, r, sigma, option_type="call"):
    """Price a European option with the Black-Scholes closed-form formula."""
    if option_type.lower() == "call" :
        sign = 1
    elif option_type.lower() == "put" : 
        sign = -1
    else : 
        raise ValueError(f"option type must be 'call' or 'put', got {option_type!r} ")
    
    if T == 0 or sigma == 0 : 
        return max(sign * (S0 - K * np.exp(-r*T) ),0)
    # Defining the threshold d2  to be in the money 
    d2 = (np.log(S0/K) + (r - (sigma**2)/2)*T)/(sigma * np.sqrt(T))
    d1 = d2 + sigma * np.sqrt(T)
    return sign * (norm.cdf(sign * d1)*S0 - K * np.exp(-r*T) * norm.cdf(sign * d2))
    
    





