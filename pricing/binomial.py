import numpy as np

def binomial_european(S0, K, T, r, sigma, n, option_type = "call"):
    "Price a European option on a recombining binomial tree. "
    h = T/n
    # High jump
    u = np.exp(sigma * (np.sqrt(h)) )
    # Down jump 
    d = 1/u
    # probability risk-free
    q = (np.exp(r*h)-d)/(u-d)
    if not (0 < q < 1):
        raise ValueError(f"Risk-neutral probability q must be strictly between 0 and 1 (no-arbitrage violated), got {q!r}")
    k = np.arange(n+1)
    S_T= S0 * (u**(k)) * (d**(n-k))
    if option_type.lower() == "call" :
        option_values = np.maximum(S_T-K,0.0)
    elif option_type.lower() == "put" :
        option_values = np.maximum(K-S_T,0.0)
    else :
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    for i in range (n-1,-1,-1):
        # we implement the recursive backward induction  from the payoff to the price of the option at t = 0 
        option_values = np.exp(-r*h)*(q * option_values[1:] + (1-q) * option_values[:-1])

    return option_values[0]





