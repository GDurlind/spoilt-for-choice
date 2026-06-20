import math

def binomial_price(spot: float, strike: float, time_to_expiry: float, rate: float, sigma: float, steps: int = 50, option_type: str = "call"):
    """
    Cox-Ross-Rubinstein binomial tree for European options.
    
    Arguments:
    spot (float): spot price / current market price of the underlying asset 
    strike (float): strike price of the option (fixed price to buy or sell)
    time_to_expiry (float): time to expiry in years
    rate (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying asset (annualized)
    steps (int): number of steps in the binomial tree
    option_type (str): "call" or "put"

    Returns:
    price (float): price of the option
    """

    # Split time, define up and down factors, and risk-neutral probabilities
    dt = time_to_expiry / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    pu = (math.exp(rate * dt) - d) / (u - d)
    pd = 1 - pu

    # Build terminal prices at maturity
    prices = [spot * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]

    # Calculate values from option types
    if option_type == "call":
        values = [max(p - strike, 0) for p in prices]
    else:
        values = [max(strike - p, 0) for p in prices]

    # Implement discounted cashflows through backward induction
    disc = math.exp(-rate * dt)
    for i in range(steps - 1, -1, -1):
        values = [disc * (pu * values[j + 1] + pd * values[j]) for j in range(i + 1)]
    return values[0]
