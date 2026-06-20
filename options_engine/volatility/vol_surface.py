import pandas as pd
import matplotlib.pyplot as plt
from ..implied.implied_vol import implied_volatility

def build_smile(df, S, r):
    """Given dataframe with columns strike, expiry, market_price, compute implied vols."""
    df = df.copy()
    df['iv'] = df.apply(lambda row: implied_volatility(row.market_price, S, row.strike, row.expiry, r), axis=1)
    return df

def plot_smile(df):
    plt.figure()
    plt.scatter(df['strike'], df['iv'])
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Smile')
    plt.grid(True)
    return plt

def plot_surface(df):
    # Simple surface via scatter colored by expiry
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['strike'], df['expiry'], df['iv'], c=df['iv'], cmap='viridis')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Expiry')
    ax.set_zlabel('Implied Vol')
    return fig
