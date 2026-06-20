import numpy as np

def long_call_payoff(S_range, K, premium=0.0):
    payoff = np.maximum(S_range - K, 0.0) - premium
    return payoff

def long_put_payoff(S_range, K, premium=0.0):
    payoff = np.maximum(K - S_range, 0.0) - premium
    return payoff

def covered_call_payoff(S_range, K, premium=0.0, long_stock=1):
    stock = long_stock * (S_range - S_range[0]) + long_stock * S_range[0]
    # assume buy 1 stock at S0, sell call
    payoff = (stock - long_stock * S_range[0]) + np.minimum(S_range - K, 0.0) + premium
    return payoff

def bull_call_spread(S_range, K1, K2, premium1=0.0, premium2=0.0):
    payoff = np.maximum(S_range - K1, 0.0) - premium1 - (np.maximum(S_range - K2, 0.0) - premium2)
    return payoff

def straddle(S_range, K, premium_call=0.0, premium_put=0.0):
    payoff = np.maximum(S_range - K, 0.0) + np.maximum(K - S_range, 0.0) - (premium_call + premium_put)
    return payoff

def strangle(S_range, K_put, K_call, premium_put=0.0, premium_call=0.0):
    payoff = np.maximum(K_put - S_range, 0.0) + np.maximum(S_range - K_call, 0.0) - (premium_put + premium_call)
    return payoff

def iron_condor(S_range, K1, K2, K3, K4, prem1=0.0, prem2=0.0, prem3=0.0, prem4=0.0):
    # Simplified payoff for long iron condor: long put K1, short put K2, short call K3, long call K4
    payoff = (
        np.maximum(K2 - S_range, 0.0) - np.maximum(K1 - S_range, 0.0)
        + np.maximum(S_range - K3, 0.0) - np.maximum(S_range - K4, 0.0)
        - (prem1 + prem2 + prem3 + prem4)
    )
    return payoff
