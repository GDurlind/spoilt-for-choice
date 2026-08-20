import pytest

from options_engine.greeks.barrier_greeks import barrier_delta, barrier_gamma, barrier_theta, barrier_vega
from options_engine.greeks.greeks import delta as bs_delta
from options_engine.greeks.greeks import gamma as bs_gamma
from options_engine.greeks.greeks import theta as bs_theta
from options_engine.greeks.greeks import vega as bs_vega
from options_engine.pricing.barrier import barrier_price_analytic
from options_engine.pricing.black_scholes import black_scholes_price

SPOT, STRIKE, BARRIER, T, RATE, SIGMA = 90.0, 100.0, 130.0, 1.0, 0.03, 0.25


def _vanilla_pricer(**kwargs):
    return black_scholes_price(kwargs["spot"], kwargs["strike"], kwargs["time_to_expiry"], kwargs["rate"], kwargs["sigma"], "call")


VANILLA_BASE = dict(spot=SPOT, strike=STRIKE, time_to_expiry=T, rate=RATE, sigma=SIGMA)


# The bump-and-revalue engine is pricer-agnostic, so it can be validated
# directly against known closed-form vanilla Greeks by feeding it the plain
# Black-Scholes pricer -- this checks the engine itself, independent of
# whether the barrier formula it's later applied to is correct.

def test_delta_matches_closed_form_on_vanilla():
    numeric = barrier_delta(_vanilla_pricer, VANILLA_BASE)
    closed_form = bs_delta(SPOT, STRIKE, T, RATE, SIGMA, "call")
    assert numeric == pytest.approx(closed_form, abs=1e-4)


def test_gamma_matches_closed_form_on_vanilla():
    numeric = barrier_gamma(_vanilla_pricer, VANILLA_BASE)
    closed_form = bs_gamma(SPOT, STRIKE, T, RATE, SIGMA)
    assert numeric == pytest.approx(closed_form, abs=1e-4)


def test_vega_matches_closed_form_on_vanilla():
    numeric = barrier_vega(_vanilla_pricer, VANILLA_BASE)
    closed_form = bs_vega(SPOT, STRIKE, T, RATE, SIGMA)
    assert numeric == pytest.approx(closed_form, abs=1e-3)


def test_theta_matches_closed_form_on_vanilla():
    numeric = barrier_theta(_vanilla_pricer, VANILLA_BASE)
    closed_form = bs_theta(SPOT, STRIKE, T, RATE, SIGMA, "call")
    assert numeric == pytest.approx(closed_form, abs=1e-3)


def test_gamma_grows_sharply_as_spot_approaches_barrier():
    # This is the actual point of building barrier Greeks: unlike a vanilla
    # option, gamma is not bounded as spot approaches the barrier -- it's why
    # hedging a barrier book gets materially harder close to the knock-out.
    far_base = dict(spot=90.0, strike=STRIKE, barrier=BARRIER, time_to_expiry=T, rate=RATE, sigma=SIGMA)
    near_base = dict(spot=129.999, strike=STRIKE, barrier=BARRIER, time_to_expiry=T, rate=RATE, sigma=SIGMA)

    # The bump must be small enough to actually straddle the tiny gap to the
    # barrier (0.001 here) -- too coarse a bump stays on the safe side of the
    # discontinuity and never sees the blow-up at all.
    gamma_far = abs(barrier_gamma(barrier_price_analytic, far_base, bump=1e-3, relative=True))
    gamma_near = abs(barrier_gamma(barrier_price_analytic, near_base, bump=1e-5, relative=True))

    assert gamma_near > 100 * gamma_far
