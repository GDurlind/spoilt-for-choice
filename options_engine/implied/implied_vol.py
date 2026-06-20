import math
from scipy.optimize import brentq
from ..pricing.black_scholes import black_scholes_price

def implied_volatility(market_price, S, K, T, r, option_type="call", tol=1e-6):
    """Solve implied volatility using Brent's method.

    Implied vol: the volatility that makes model price equal market price.
    Traders use it to compare relative option value across strikes/expiries.
    """
    def diff(sigma):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        return price - market_price

    # vol bounds
    low, high = 1e-6, 5.0
    try:
        iv = brentq(diff, low, high, xtol=tol)
    except Exception:
        return float('nan')
    return iv
