import yfinance as yf

tickers = ["AMZN", "AON", "AXP", "AAPL"]

for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        data = t.history(period="1mo")
        print(f"{ticker}: {len(data)}행")
        if len(data) > 0:
            print(f"  최근 날짜: {data.index[-1]}")
            print(f'  최근 종가: ${data["Close"].iloc[-1]:.2f}')
    except Exception as e:
        print(f"{ticker}: 오류 - {e}")
    print()
