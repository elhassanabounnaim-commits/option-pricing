import numpy as np
from typing import NamedTuple

class MC(NamedTuple) :
    price : float
    std_error : float
    ci : tuple[float,float]

def monte_carlo_european(S0, K, T, r, sigma, n_paths,option_type="call", seed=None, antithetic=False,control_variate=False):
    """Price a European option by Monte Carlo simulation.

    Returns the discounted price, its standard error and a 95% confidence
    interval.
    """

    if option_type.lower() == "call" : 
        sign = 1
    elif option_type.lower() == "put" : 
        sign = -1 
    else : 
        raise ValueError(f"option type must be 'call' or 'put', got {option_type!r} ")
    
    rng  = np.random.default_rng(seed)
    # On génère les réalisations de la loi normale 
    if antithetic :
        n_base = n_paths // 2
        z = rng.standard_normal(n_base)
        Z = np.concatenate([z,-z])
    else : 
        Z = rng.standard_normal(n_paths)
    

    # Le vecteur des prix de sous jacent selon les N trajectoires 
    S_T = S0 * np.exp((r - (sigma**2)/2)*T + sigma * np.sqrt(T)* Z)

    # Vecteur des payoff actualisés
    X = np.exp(-r * T) * np.maximum(sign* (S_T - K),0.0)
    if control_variate : 
        ES = np.exp(r * T) * S0
        cov  = np.cov(X, S_T, ddof=1)
        c = cov[0,1]/cov[1,1]
        X = X - c * (S_T - ES)
    if antithetic : 
        sample = (X[:n_base] + X[n_base:]) / 2
    else : 
        sample = X

    
    # la valeur estimé par monte carlo 
    price = float(sample.mean())
    std_dev = float(sample.std(ddof = 1))
    std_error  = float(std_dev / np.sqrt(sample.size))

    # les deux bornes de l'intervalle de confiance à 95 % 
    C_min = price - 1.96 * std_error
    C_max = price + 1.96 * std_error

    return MC(price, std_error, (C_min,C_max))




    

    



