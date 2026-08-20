def _bumped(base_kwargs, arg, value):
    kwargs = dict(base_kwargs)
    kwargs[arg] = value
    return kwargs


def _step(x0, bump, relative):
    return x0 * bump if relative else bump


def barrier_delta(pricer, base_kwargs: dict, bump: float = 1e-3, relative: bool = True) -> float:
    """
    First derivative of price with respect to spot, by central finite difference.

    `pricer` is any barrier pricing callable -- barrier_price_analytic or
    barrier_price_mc -- that accepts the same keyword arguments as
    `base_kwargs`. Barrier options don't generally have clean closed-form
    Greeks once you leave the simplest single-barrier case, so this reproduces
    them numerically from whatever pricer is on hand instead.

    `base_kwargs` (including `seed`, when the pricer is barrier_price_mc) is
    reused unchanged for both the up and down revaluation, so both draws see
    identical random numbers and only `spot` differs between them. This
    "common random numbers" property is what keeps a Monte Carlo finite
    difference from being swamped by independent simulation noise -- without
    it, gamma in particular becomes unusable.

    Returns:
    delta (float): d(price)/d(spot)
    """
    spot = base_kwargs["spot"]
    h = _step(spot, bump, relative)
    price_up = pricer(**_bumped(base_kwargs, "spot", spot + h))
    price_down = pricer(**_bumped(base_kwargs, "spot", spot - h))
    return (price_up - price_down) / (2 * h)


def barrier_gamma(pricer, base_kwargs: dict, bump: float = 1e-2, relative: bool = True) -> float:
    """
    Second derivative of price with respect to spot, by central finite difference.

    Second differences amplify noise far more than first differences, so this
    uses a larger default bump than barrier_delta -- and is exactly where
    common random numbers (see barrier_delta) matter most when pricer is
    barrier_price_mc.

    Returns:
    gamma (float): d^2(price)/d(spot)^2
    """
    spot = base_kwargs["spot"]
    h = _step(spot, bump, relative)
    price_up = pricer(**_bumped(base_kwargs, "spot", spot + h))
    price_mid = pricer(**base_kwargs)
    price_down = pricer(**_bumped(base_kwargs, "spot", spot - h))
    return (price_up - 2 * price_mid + price_down) / (h ** 2)


def barrier_vega(pricer, base_kwargs: dict, bump: float = 1e-3, relative: bool = True) -> float:
    """
    First derivative of price with respect to volatility, by central finite difference.

    Returns:
    vega (float): d(price)/d(sigma)
    """
    sigma = base_kwargs["sigma"]
    h = _step(sigma, bump, relative)
    price_up = pricer(**_bumped(base_kwargs, "sigma", sigma + h))
    price_down = pricer(**_bumped(base_kwargs, "sigma", sigma - h))
    return (price_up - price_down) / (2 * h)


def barrier_theta(pricer, base_kwargs: dict, bump: float = 1 / 365, relative: bool = False) -> float:
    """
    Price decay per year of calendar time passing, i.e. -d(price)/d(time_to_expiry),
    matching the sign convention of greeks.theta (negative for a typical long option).

    Falls back to a one-sided difference when time_to_expiry is too close to zero
    for a symmetric bump to stay in the option's valid domain.

    Returns:
    theta (float): decay in price per year of calendar time
    """
    T = base_kwargs["time_to_expiry"]
    h = _step(T, bump, relative)
    price_up = pricer(**_bumped(base_kwargs, "time_to_expiry", T + h))
    if T - h > 0:
        price_down = pricer(**_bumped(base_kwargs, "time_to_expiry", T - h))
        return -(price_up - price_down) / (2 * h)
    price_now = pricer(**base_kwargs)
    return -(price_up - price_now) / h
