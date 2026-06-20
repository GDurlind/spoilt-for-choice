import streamlit as st
from options_engine.pricing.black_scholes import black_scholes_price
from options_engine.implied.implied_vol import implied_volatility
from options_engine.greeks.greeks import delta, gamma, vega, theta, rho

st.sidebar.title('Options Analytics')
page = st.sidebar.selectbox('Page', ['Pricing', 'Implied Vol', 'Greeks'])

if page == 'Pricing':
    S = st.sidebar.number_input('Spot', value=100.0)
    K = st.sidebar.number_input('Strike', value=100.0)
    T = st.sidebar.number_input('Time to expiry (years)', value=30/365)
    r = st.sidebar.number_input('Risk-free rate', value=0.01)
    sigma = st.sidebar.number_input('Volatility', value=0.2)
    call = black_scholes_price(S, K, T, r, sigma, 'call')
    put = black_scholes_price(S, K, T, r, sigma, 'put')
    st.write('Black-Scholes Call:', call)
    st.write('Black-Scholes Put:', put)

elif page == 'Implied Vol':
    S = st.sidebar.number_input('Spot', value=100.0)
    K = st.sidebar.number_input('Strike', value=100.0)
    T = st.sidebar.number_input('Time to expiry (years)', value=30/365)
    r = st.sidebar.number_input('Risk-free rate', value=0.01)
    market = st.sidebar.number_input('Market price', value=2.0)
    opt = st.sidebar.selectbox('Type', ['call', 'put'])
    iv = implied_volatility(market, S, K, T, r, opt)
    st.write('Market price:', market)
    st.write('Implied volatility:', iv)

else:
    S = st.sidebar.number_input('Spot', value=100.0)
    K = st.sidebar.number_input('Strike', value=100.0)
    T = st.sidebar.number_input('Time to expiry (years)', value=30/365)
    sigma = st.sidebar.number_input('Volatility', value=0.2)
    opt = st.sidebar.selectbox('Type', ['call', 'put'])
    st.write('Delta:', delta(S, K, T, 0.01, sigma, opt))
    st.write('Gamma:', gamma(S, K, T, 0.01, sigma))
    st.write('Vega:', vega(S, K, T, 0.01, sigma))
    st.write('Theta:', theta(S, K, T, 0.01, sigma, opt))
    st.write('Rho:', rho(S, K, T, 0.01, sigma, opt))
