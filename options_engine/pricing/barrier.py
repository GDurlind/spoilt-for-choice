import math

import numpy as np
from scipy.stats import norm

# Broadie-Glasserman-Kou (1997) continuity correction constant: -zeta(1/2) / sqrt(2*pi).
# A continuously-monitored closed form and a discretely-monitored simulation price
# different things -- checking the barrier only at `steps` points lets a path spike
# through and back undetected, so discrete monitoring is systematically worth more.
# Shifting the barrier outward by this factor makes the closed form approximate the
# discretely-monitored price instead, so the two can be compared on equal terms.
_BGK_BETA = 0.5826


def barrier_price_analytic(spot: float, strike: float, barrier: float, time_to_expiry: float, rate: float, sigma: float, monitoring_steps: int = None) -> float:
    """
    Closed-form price for a continuously-monitored up-and-out call, no rebate.

    Reiner-Rubinstein (1991) standard barrier formula, specialised to a call
    struck below the barrier (phi=1, eta=-1 in their general notation). Assumes
    no continuous dividend yield, so cost of carry b = rate.

    Arguments:
    spot (float): spot price / current market price of the underlying asset
    strike (float): strike price of the option (fixed price to buy or sell)
    barrier (float): knock-out level; the option is worthless once spot reaches it
    time_to_expiry (float): time to expiry in years
    rate (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying asset (annualized)
    monitoring_steps (int): if set, apply the Broadie-Glasserman-Kou correction so
        this continuous-monitoring formula approximates a barrier checked only at
        this many discrete points -- i.e. what barrier_price_mc with the same
        `steps` converges to. Leave as None for the true continuous-monitoring price.

    Returns:
    price (float): price of the up-and-out call
    """
    if spot >= barrier:
        # Already touched (or through) the barrier: dead on arrival.
        return 0.0
    if barrier <= strike:
        # Any path that could finish in the money must first pass through the
        # barrier on the way there, so the option can never pay off.
        return 0.0
    if time_to_expiry <= 0 or sigma <= 0:
        return max(spot - strike, 0.0)

    effective_barrier = barrier
    if monitoring_steps is not None:
        dt = time_to_expiry / monitoring_steps
        effective_barrier = barrier * math.exp(_BGK_BETA * sigma * math.sqrt(dt))
        if spot >= effective_barrier:
            return 0.0

    b = rate  # cost of carry; no continuous dividend yield in this engine
    sig_sqrt_t = sigma * math.sqrt(time_to_expiry)
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2

    x1 = math.log(spot / strike) / sig_sqrt_t + (1 + mu) * sig_sqrt_t
    x2 = math.log(spot / effective_barrier) / sig_sqrt_t + (1 + mu) * sig_sqrt_t
    y1 = math.log(effective_barrier ** 2 / (spot * strike)) / sig_sqrt_t + (1 + mu) * sig_sqrt_t
    y2 = math.log(effective_barrier / spot) / sig_sqrt_t + (1 + mu) * sig_sqrt_t

    disc_spot = spot * math.exp((b - rate) * time_to_expiry)
    disc_strike = strike * math.exp(-rate * time_to_expiry)
    h_ratio = effective_barrier / spot

    # A is the plain Black-Scholes call price; B, C, D are barrier correction
    # terms. A - B + C - D collapses to A (vanilla) as the barrier moves to
    # infinity, and to 0 as the barrier moves down to the strike -- both are
    # good sanity checks to run against black_scholes_price.
    A = disc_spot * norm.cdf(x1) - disc_strike * norm.cdf(x1 - sig_sqrt_t)
    B = disc_spot * norm.cdf(x2) - disc_strike * norm.cdf(x2 - sig_sqrt_t)
    C = (disc_spot * h_ratio ** (2 * (mu + 1)) * norm.cdf(-y1)
         - disc_strike * h_ratio ** (2 * mu) * norm.cdf(-y1 + sig_sqrt_t))
    D = (disc_spot * h_ratio ** (2 * (mu + 1)) * norm.cdf(-y2)
         - disc_strike * h_ratio ** (2 * mu) * norm.cdf(-y2 + sig_sqrt_t))

    return A - B + C - D


def barrier_price_mc(spot: float, strike: float, barrier: float, time_to_expiry: float, rate: float, sigma: float, option_type: str = "call", sims: int = 10000, steps: int = 252, seed: int = None) -> float:
    """
    Monte Carlo price for an up-and-out barrier option, no rebate.

    Simulates full spot price paths (not just the terminal price) since the
    option is extinguished the moment the path touches or crosses the barrier,
    at any point before expiry. Continuous monitoring is approximated by
    checking the barrier at each simulated step; increasing `steps` tightens
    that approximation towards barrier_price_analytic's continuous price (see
    the BGK correction there for comparing the two at matched monitoring
    frequency).

    Arguments:
    spot (float): spot price / current market price of the underlying asset
    strike (float): strike price of the option (fixed price to buy or sell)
    barrier (float): knock-out level; the option pays 0 if spot ever reaches it
    time_to_expiry (float): time to expiry in years
    rate (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying asset (annualized)
    option_type (str): "call" or "put"
    sims (int): number of Monte Carlo simulations
    steps (int): number of time steps per path (monitoring frequency)
    seed (int): random seed for reproducibility

    Returns:
    price (float): Monte Carlo price of the up-and-out barrier option
    """
    if spot >= barrier:
        return 0.0
    if time_to_expiry <= 0 or sigma <= 0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    rng = np.random.default_rng(seed)
    dt = time_to_expiry / steps

    # Draw every step of every path in one shot, then cumulatively sum
    # log-returns along the time axis so column j holds each path's
    # log-return from t=0 up to step j.
    Z = rng.standard_normal((sims, steps))
    log_returns = (rate - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = spot * np.exp(log_paths)

    breached = np.any(paths >= barrier, axis=1)

    terminal = paths[:, -1]
    if option_type == "call":
        payoffs = np.maximum(terminal - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal, 0.0)
    payoffs = np.where(breached, 0.0, payoffs)

    price = math.exp(-rate * time_to_expiry) * np.mean(payoffs)
    return price
