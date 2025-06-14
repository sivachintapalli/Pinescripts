import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

symbol = 'SPY'
period = '7d'  # yfinance only allows up to ~7 days for 1 minute data
interval = '1m'

print('Downloading data...')
df = yf.download(symbol, period=period, interval=interval, progress=False)
if df.empty:
    raise SystemExit('Failed to download data; yfinance returned empty DataFrame')

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Ensure the index is timezone-aware for plotting
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

df['hlc3'] = (df['High'] + df['Low'] + df['Close']) / 3

AP = 14
coeff = 1.0

# Money Flow Index calculation
typical = df['hlc3']
money_flow = typical * df['Volume']
up_mf = money_flow.where(typical > typical.shift(1), 0)
down_mf = money_flow.where(typical < typical.shift(1), 0)
pos_sum = up_mf.rolling(window=AP).sum()
neg_sum = down_mf.rolling(window=AP).sum()
mfi = 100 - 100 / (1 + pos_sum / neg_sum)

# True range and ATR
tr = df[['High','Low','Close']].copy()
tr['prev_close'] = tr['Close'].shift(1)
tr['tr'] = tr[['High','prev_close']].max(axis=1) - tr[['Low','prev_close']].min(axis=1)
ATR = tr['tr'].rolling(window=AP).mean()

upT = df['Low'] - ATR * coeff
downT = df['High'] + ATR * coeff

AlphaTrend = np.zeros(len(df))

for i in range(len(df)):
    mfi_cond = mfi.iloc[i] >= 50
    use_up = bool(mfi_cond)
    prev = AlphaTrend[i-1] if i>0 else 0
    if use_up:
        AlphaTrend[i] = prev if upT.iloc[i] < prev else upT.iloc[i]
    else:
        AlphaTrend[i] = prev if downT.iloc[i] > prev else downT.iloc[i]

df['AlphaTrend'] = AlphaTrend
# Signals: cross of AlphaTrend with its value 2 bars back
cross_up = (df['AlphaTrend'] > df['AlphaTrend'].shift(2)) & (df['AlphaTrend'].shift(1) <= df['AlphaTrend'].shift(3))
cross_down = (df['AlphaTrend'] < df['AlphaTrend'].shift(2)) & (df['AlphaTrend'].shift(1) >= df['AlphaTrend'].shift(3))
signal = np.where(cross_up, 1, np.where(cross_down, -1, 0))

df['signal'] = signal
# Forward fill signal for direction
position = pd.Series(signal).replace(0, np.nan).ffill().fillna(0)

df['position'] = position

# Simple backtest: assume 1 share, compute returns
df['returns'] = df['Close'].pct_change()
df['strategy'] = df['returns'] * df['position'].shift(1)

equity = (1 + df['strategy']).cumprod()

plt.figure(figsize=(10,6))
plt.plot(df.index, df['Close'], label='SPY')
plt.plot(df.index, df['AlphaTrend'], label='AlphaTrend')
plt.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1], marker='^', color='g', label='Buy')
plt.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1], marker='v', color='r', label='Sell')
plt.legend()
plt.title('AlphaTrend on SPY ({} {})'.format(period, interval))
plt.tight_layout()
plt.savefig('alpha_trend_backtest.png')

plt.figure()
plt.plot(equity)
plt.title('Equity Curve')
plt.tight_layout()
plt.savefig('equity_curve.png')
print('Backtest complete. Results saved to alpha_trend_backtest.png and equity_curve.png')
