import numpy as np
from ..greeks.greeks import delta, gamma, vega, theta

def simple_pnl_attribution(position, S0, S_paths, T0, T_paths, r, sigma):
    """Estimate daily PnL contributions from Greeks using first-order approximations.

    position: dict with keys 'type'('call'/'put'), 'K', 'qty', 'sigma'
    S_paths: array of spot prices over time
    T_paths: array of times to expiry corresponding to S_paths
    Returns breakdown arrays and total PnL
    """
    qty = position.get('qty', 1)
    K = position['K']
    opt = position.get('type', 'call')

    pnl_delta = []
    pnl_gamma = []
    pnl_theta = []
    pnl_vega = []

    prev_price = None
    for i in range(len(S_paths)):
        S = S_paths[i]
        T = T_paths[i]
        if T <= 0:
            break
        d = delta(S, K, T, r, position['sigma'], opt)
        g = gamma(S, K, T, r, position['sigma'])
        v = vega(S, K, T, r, position['sigma'])
        th = theta(S, K, T, r, position['sigma'], opt)

        if prev_price is None:
            prev_price = S
            pnl_delta.append(0.0)
            pnl_gamma.append(0.0)
            pnl_theta.append(th * 1)
            pnl_vega.append(0.0)
            continue

        dS = S - prev_price
        pnl_delta.append(d * dS * qty)
        pnl_gamma.append(0.5 * g * (dS ** 2) * qty)
        pnl_theta.append(th * 1 * qty)
        # assume small vol change proportional to 1/S
        dv = 0.01 * position['sigma']
        pnl_vega.append(v * dv * qty)

        prev_price = S

    total = np.sum(pnl_delta) + np.sum(pnl_gamma) + np.sum(pnl_theta) + np.sum(pnl_vega)
    return {
        'delta': np.array(pnl_delta),
        'gamma': np.array(pnl_gamma),
        'theta': np.array(pnl_theta),
        'vega': np.array(pnl_vega),
        'total': total,
    }
