import yfinance as yf

# As of 2024-06-13. Dropped 'GEHC' since yfinance does not have data for it. Dropped 'GFS', 'CEG', 'DASH' and 'ABNB' because of a lack of old data.
nasdaq = ["MSFT", "NVDA", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "COST", "TSLA", "NFLX", "AMD", "QCOM", "PEP", "TMUS", "LIN", "ADBE", "AMAT", "CSCO", "TXN", "AMGN", "INTU", "CMCSA", "MU", "ISRG", "HON", "LRCX", "INTC", "BKNG", "VRTX", "ADI", "REGN", "KLAC", "ADP", "PANW", "PDD", "SBUX", "MDLZ", "ASML", "SNPS", "CRWD", "GILD", "MELI", "CDNS", "PYPL", "NXPI", "CTAS", "MAR", "CSX", "MRVL", "ROP", "ORLY", "MRNA", "PCAR", "MNST", "CPRT", "MCHP", "ROST", "KDP", "AZN", "ADSK", "AEP", "FTNT", "DXCM", "WDAY", "PAYX", "IDXX", "TTD", "KHC", "CHTR", "LULU", "VRSK", "ODFL", "EA", "FAST", "EXC", "FANG", "DDOG", "CCEP", "CTSH", "BIIB", "BKR", "ON", "CSGP", "XEL", "CDW", "ANSS", "TTWO", "ZS", "TEAM", "DLTR", "WBD", "ILMN", "MDB", "WBA", "SIRI"]

# Download historical stock data, keep monthly adjusted close price, reorder to match the standard order
# We use monthly data because the returns are less noisy than daily data
price_data = yf.download(nasdaq, start="2020-01-01", end="2022-12-31", interval="1mo")['Adj Close'][nasdaq]

assert(price_data.isna().any().any() == 0)

# Calculate monthly returns
monthly_returns = price_data.pct_change().dropna()
print(monthly_returns)
# Save the data
monthly_returns.to_csv('../data/monthly_returns.csv')