import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Smart Investment Portfolio Management System", layout="wide")

THEMES = {
    "Light": {"bgcolor": "#FFFFFF", "font_color": "#000000"},
    "Dark": {"bgcolor": "#0E1117", "font_color": "#FFFFFF"}
}
theme = st.sidebar.selectbox("Choose theme", options=list(THEMES.keys()), index=0)
bgcolor = THEMES[theme]["bgcolor"]
font_color = THEMES[theme]["font_color"]

st.title("📈 Smart Investment Portfolio Management System 🇮🇳")

budget = st.sidebar.number_input("Investment Amount (₹)", value=100000, step=1000, min_value=1000,
                                 help="Set your total investable budget")
risk_pref = st.sidebar.selectbox("Risk Preference", options=["low", "medium", "high"],
                                 help="Choose your risk tolerance level")

stocks = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "LT.NS": "Larsen & Toubro",
    "SBIN.NS": "State Bank of India",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "BHARTIARTL.NS": "Bharti Airtel",
    "MARUTI.NS": "Maruti Suzuki"
}

@st.cache_data(ttl=3600)
def get_data(tickers):
    df = yf.download(tickers, period="6mo", interval="1d", progress=False)["Close"]
    return df

data = get_data(list(stocks.keys()))
latest_prices = data.iloc[-1]
returns = data.pct_change().mean() * 100

risk_factor = {"low": 0.5, "medium": 1.0, "high": 1.5}[risk_pref]
adjusted_returns = returns * risk_factor

items = []
for ticker in stocks.keys():
    price = latest_prices[ticker]
    ret = adjusted_returns[ticker]
    ratio = ret / price if price > 0 else 0
    items.append((ticker, stocks[ticker], price, ret, ratio))
items.sort(key=lambda x: x[4], reverse=True)

remaining_budget = budget
portfolio = []

for ticker, name, price, ret, ratio in items:
    if price <= 0 or remaining_budget <= 0:
        continue
    fraction = min(1.0, remaining_budget / price)
    cost = fraction * price
    exp_return = fraction * ret
    portfolio.append((ticker, name, price, fraction, cost, exp_return))
    remaining_budget -= cost

total_cost = sum(x[4] for x in portfolio)
scale = budget / total_cost if total_cost else 1.0

portfolio_scaled = []
for ticker, name, price, fraction, cost, exp_return in portfolio:
    fraction_scaled = fraction * scale
    cost_scaled = fraction_scaled * price
    exp_return_scaled = exp_return * scale
    portfolio_scaled.append((ticker, name, price, fraction_scaled, cost_scaled, exp_return_scaled))

df_portfolio = pd.DataFrame(portfolio_scaled, columns=["Ticker", "Stock", "Unit Price (₹)", "Fraction", "Cost (₹)", "Expected Return (%)"])
df_portfolio["Fraction"] = df_portfolio["Fraction"].round(4)
df_portfolio["Cost (₹)"] = df_portfolio["Cost (₹)"].round(2)
df_portfolio["Expected Return (%)"] = df_portfolio["Expected Return (%)"].round(2)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Optimal Fractional Portfolio Allocation")
    st.dataframe(df_portfolio[["Stock", "Unit Price (₹)", "Fraction", "Cost (₹)", "Expected Return (%)"]])
    st.write(f"Total Invested: ₹{df_portfolio['Cost (₹)'].sum():,.2f} / Budget: ₹{budget:,}")
    st.write(f"Estimated Portfolio Daily Return: {df_portfolio['Expected Return (%)'].sum():.2f} %")

    fig1 = px.bar(df_portfolio, x="Stock", y=df_portfolio["Expected Return (%)"] * df_portfolio["Fraction"],
                  labels={"y": "Weighted Expected Return (%)"}, title="Expected Returns Weighted by Allocation",
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.header("Benchmark & Risk Metrics")

    nifty = yf.download("^NSEI", period="1mo", interval="1d", progress=False)["Close"]
    if not nifty.empty and len(nifty) > 1:
        nifty_return = ((nifty.iloc[-1] / nifty.iloc[0]) - 1) * 100
        show_nifty_return = f"{float(nifty_return):.2f}%"
    else:
        show_nifty_return = "N/A"

    st.metric("NIFTY 50 1mo Return", show_nifty_return)
    port_return = df_portfolio["Expected Return (%)"].sum()
    st.metric("Portfolio Expected Return", f"{port_return:.2f}%")

    hist_returns = data.pct_change().fillna(0)
    weighted_returns = hist_returns.mul(df_portfolio.set_index("Ticker")["Fraction"], axis=1).sum(axis=1)
    cum_returns = (1 + weighted_returns).cumprod() - 1

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns.values * 100, mode="lines", name="Portfolio"))

    if show_nifty_return != "N/A":
        nifty_cum = (1 + nifty.pct_change().fillna(0)).cumprod() - 1
        fig2.add_trace(go.Scatter(x=nifty_cum.index, y=nifty_cum.values * 100, mode="lines", name="NIFTY 50 Benchmark"))

    fig2.update_layout(
        title="Cumulative Returns (%)",
        xaxis_title="Date",
        yaxis_title="Returns (%)",
        plot_bgcolor=THEMES[theme]["bgcolor"],
        paper_bgcolor=THEMES[theme]["bgcolor"],
        font_color=THEMES[theme]["font_color"],
    )
    st.plotly_chart(fig2, use_container_width=True)

    vol = weighted_returns.std() * np.sqrt(252)
    sharpe = (weighted_returns.mean() * 252) / vol if vol > 0 else float("nan")

    st.write(f"Annualized Volatility: {vol:.2%}")
    st.write(f"Sharpe Ratio: {sharpe:.2f}")

# Correlation Heatmap
st.subheader("Correlation Heatmap of Stocks")
daily_returns = data.pct_change().dropna()
fig_corr, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(daily_returns.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig_corr)

# Returns Distribution
st.subheader("Distribution of Daily Returns per Stock")
fig_dist = px.violin(daily_returns, box=True, points="all", color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig_dist, use_container_width=True)

# Monthly average returns
monthly_returns = data.pct_change().resample("M").mean().mean(axis=1) * 100
st.subheader("Average Monthly Returns")
fig_monthly = px.bar(
    x=monthly_returns.index.strftime("%b %Y"),
    y=monthly_returns.values,
    labels={"x": "Month", "y": "Average Monthly Return (%)"},
    color=monthly_returns.values,
    color_continuous_scale=px.colors.sequential.Viridis,
)
st.plotly_chart(fig_monthly, use_container_width=True)

# Export CSV
csv = df_portfolio.to_csv(index=False).encode()
st.download_button("Download Portfolio CSV", data=csv, file_name="portfolio.csv", mime="text/csv")

# Stock profiles
st.header("Stock Profiles")
for ticker, name in stocks.items():
    st.markdown(f"**{name}** - [Yahoo Finance](https://finance.yahoo.com/quote/{ticker})")

st.markdown("---")
st.markdown("Programmed by Shivam Sharma, Gaurav Mehra, Avikam Rana and Ansh Tripathi")