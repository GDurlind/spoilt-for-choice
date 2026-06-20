import math
from scipy.stats import norm

def black_scholes_price(spot: float, strike: float, time_to_expiry: float, rate: float, sigma: float, option_type: str = "call") -> float:
    """
    Return Black-Scholes price for European call/put.

    Arguments:
    spot (float): spot price / current market price of the underlying asset 
    strike (float): strike price of the option (fixed price to buy or sell)
    time_to_expiry (float): time to expiry in years
    rate (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying asset (annualized)
    option_type (str): "call" or "put"
    
    Returns:
    price (float): Black-Scholes price of the option
    """

    if time_to_expiry <= 0 or sigma <= 0:
        # At expiry or zero volatility set intrinsic value
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    # Calculate option's delta and expectation of underlying asset in the payoff
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * math.sqrt(time_to_expiry))
    
    # Calculate risk neutral probability option finishes in the money
    d2 = d1 - sigma * math.sqrt(time_to_expiry)

    # Assign price relative to option type
    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)
    else:
        price = strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return price
