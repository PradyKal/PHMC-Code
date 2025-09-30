import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor

plt.rcParams['figure.figsize'] = (12,6)

def generate_synthetic_price(n=2000, seed=42):
    np.random.seed(seed)
    dt = 1/252
    mu = 0.0002
    sigma = 0.01
    returns = np.random.normal(loc=mu, scale=sigma, size=n)
    # insert trending segments
    for start in [200, 600, 1100, 1500]:
        length = np.random.randint(40, 120)
        returns[start:start+length] += np.abs(np.random.normal(0.0015, 0.0008, size=length))
    price = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.Series(price, index=dates, name='close')

def atr(high, low, close, n=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=n, min_periods=1).mean()
    return atr

def compute_run_lengths(returns):
    signs = (returns > 0).astype(int)
    rl = np.zeros(len(signs), dtype=int)
    run_length = 0
    for i, s in enumerate(signs):
        if s == 1:
            run_length += 1
        else:
            run_length = 0
        rl[i] = run_length
    return pd.Series(rl, index=returns.index)

def topo_trend_from_runs(run_lengths, lookback=100, min_run_length=3):
    idx = run_lengths.index
    out = pd.Series(0.0, index=idx)
    rl = run_lengths.values
    n = len(rl)
    run_ends = []
    for i in range(n):
        if rl[i] > 0 and (i==n-1 or rl[i+1]==0):
            length = rl[i]
            run_ends.append((i, length))
    lb = lookback
    run_ends_idx = np.array([e[0] for e in run_ends])
    run_ends_len = np.array([e[1] for e in run_ends])
    for t in range(n):
        left = t - lb + 1
        if left < 0:
            left = 0
        mask = (run_ends_idx >= left) & (run_ends_idx <= t)
        if np.any(mask):
            lengths = run_ends_len[mask]
            lengths = lengths[lengths >= min_run_length]
            out.iloc[t] = lengths.sum() if len(lengths)>0 else 0.0
        else:
            out.iloc[t] = 0.0
    return out

def gini_coefficient(x):
    """
    Calculate Gini coefficient for a series.
    Returns value between 0 (perfect equality) and 1 (perfect inequality).
    Works with positive values; for returns we use absolute values.
    """
    x = np.array(x)
    x = np.abs(x)  # Use absolute values for returns
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

def rolling_gini(series, window):
    """Calculate rolling Gini coefficient."""
    return series.rolling(window=window, min_periods=max(5, window//2)).apply(
        lambda x: gini_coefficient(x), raw=True
    )

def gini_adjusted_sizing(base_position, gini_score, gini_threshold=0.65, scale_factor=0.5):
    """
    Reduce position size when Gini is high (returns too concentrated).
    When gini > threshold, scale down position.
    """
    if gini_score > gini_threshold:
        penalty = 1 - scale_factor * ((gini_score - gini_threshold) / (1 - gini_threshold))
        penalty = max(0.1, penalty)  # Keep at least 10% of position
        return base_position * penalty
    return base_position

def compute_signals(df, params):
    close = df['close']
    high = df.get('high', close)
    low = df.get('low', close)
    returns = close.pct_change().fillna(0)
    
    df['returns'] = returns
    df['run_len_pos'] = compute_run_lengths(returns)
    df['atr'] = atr(high, low, close, n=params['atr_period']).replace(0, np.nan).fillna(method='bfill')
    df['topo_raw'] = topo_trend_from_runs(df['run_len_pos'], lookback=params['lookback'], 
                                          min_run_length=params['min_run_len'])
    df['topo_trend'] = df['topo_raw'] / df['atr']
    
    # Rolling Gini coefficient on returns
    df['gini_returns'] = rolling_gini(df['returns'], window=params['gini_window'])
    df['gini_returns'] = df['gini_returns'].fillna(method='bfill').fillna(0.5)
    
    # Rolling Gini on strategy returns (computed later, so we'll update this)
    df['topo_mu'] = df['topo_trend'].rolling(window=params['z_window'], min_periods=5).mean()
    df['topo_sigma'] = df['topo_trend'].rolling(window=params['z_window'], min_periods=5).std().replace(0, np.nan).fillna(method='bfill')
    df['Z'] = (df['topo_trend'] - df['topo_mu']) / df['topo_sigma']
    
    # Gini-adjusted entry threshold: increase threshold when market Gini is high
    df['entry_threshold_adj'] = params['entry_z'] * (1 + params['gini_adjustment'] * df['gini_returns'])
    df['entry'] = (df['Z'] > df['entry_threshold_adj']).astype(int)
    
    # Base position sizing
    raw_pos = params['size_scale'] * df['Z'] / df['atr']
    df['position_base'] = raw_pos.clip(lower=0, upper=params['max_pos']) * df['entry']
    
    # Apply Gini-based position adjustment
    df['position'] = df.apply(
        lambda row: gini_adjusted_sizing(row['position_base'], row['gini_returns'], 
                                        params['gini_threshold'], params['gini_scale']),
        axis=1
    )
    
    # Forward-fill and exit logic
    df['position'] = df['position'].where(df['entry'] == 1)
    df['position'] = df['position'].ffill().fillna(0)
    df.loc[df['Z'] < params['exit_z'], 'position'] = 0
    
    # Strategy returns
    df['strat_ret'] = df['position'].shift(1) * df['returns']
    df['strat_ret'] = df['strat_ret'].fillna(0)
    
    # Rolling Gini on strategy returns (for monitoring)
    df['gini_strategy'] = rolling_gini(df['strat_ret'], window=params['gini_window'])
    df['gini_strategy'] = df['gini_strategy'].fillna(method='bfill').fillna(0.5)
    
    # Performance
    df['cum_strat'] = (1 + df['strat_ret']).cumprod()
    df['cum_buyhold'] = (1 + df['returns']).cumprod()
    
    return df

def backtest_with_params(price_series, params):
    df = pd.DataFrame({'close': price_series})
    df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.002, size=len(df))))
    df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.002, size=len(df))))
    df = compute_signals(df, params)
    return df

def plot_lorenz_curve(returns, strategy_returns, title="Lorenz Curve"):
    """Plot Lorenz curves for returns distribution."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (ret, label) in enumerate([(returns, 'Market Returns'), 
                                         (strategy_returns, 'Strategy Returns')]):
        ret_abs = np.abs(ret[ret != 0])
        if len(ret_abs) == 0:
            continue
        sorted_ret = np.sort(ret_abs)
        cumsum = np.cumsum(sorted_ret)
        cumsum_pct = cumsum / cumsum[-1]
        n = len(sorted_ret)
        lorenz_x = np.arange(1, n+1) / n
        
        gini = gini_coefficient(ret_abs)
        
        ax[idx].plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=1)
        ax[idx].plot(lorenz_x, cumsum_pct, 'b-', linewidth=2, label=f'Lorenz Curve\nGini = {gini:.3f}')
        ax[idx].fill_between(lorenz_x, cumsum_pct, lorenz_x, alpha=0.3)
        ax[idx].set_xlabel('Cumulative Share of Periods')
        ax[idx].set_ylabel('Cumulative Share of Returns')
        ax[idx].set_title(f'{label}\nGini Coefficient: {gini:.3f}')
        ax[idx].legend()
        ax[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Enhanced parameters with Gini modeling
params = {
    'atr_period': 14,
    'lookback': 120,
    'min_run_len': 3,
    'z_window': 100,
    'entry_z': 1.5,
    'exit_z': 0.7,
    'size_scale': 0.5,
    'max_pos': 2.0,
    # Gini parameters
    'gini_window': 60,           # Rolling window for Gini calculation
    'gini_threshold': 0.65,      # Threshold above which to reduce position
    'gini_scale': 0.5,           # How much to scale down when Gini is high
    'gini_adjustment': 0.3       # How much to increase entry threshold based on Gini
}

price = generate_synthetic_price(n=2000)
df = backtest_with_params(price, params)

# Performance summary
total_days = len(df)
trades = ((df['position'].diff() != 0) & (df['position'] > 0)).sum()
ann_return_strat = (df['cum_strat'].iloc[-1]) ** (252/total_days) - 1
ann_return_bh = (df['cum_buyhold'].iloc[-1]) ** (252/total_days) - 1

# Gini statistics
final_gini_market = gini_coefficient(df['returns'].dropna())
final_gini_strategy = gini_coefficient(df['strat_ret'][df['strat_ret'] != 0].dropna())

print("=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)
print(f"Trades opened: {trades}")
print(f"Strategy cumulative return: {df['cum_strat'].iloc[-1]:.3f} | Annualized ~ {ann_return_strat:.2%}")
print(f"Buy & Hold cumulative return: {df['cum_buyhold'].iloc[-1]:.3f} | Annualized ~ {ann_return_bh:.2%}")
print(f"\nGINI COEFFICIENTS (Return Concentration)")
print(f"Market Gini: {final_gini_market:.3f}")
print(f"Strategy Gini: {final_gini_strategy:.3f}")
print(f"Interpretation: Lower Gini = more distributed returns (better)")
print("=" * 60)

# Plot 1: Cumulative Performance with Gini overlay
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax[0].plot(df.index, df['cum_buyhold'], label='Buy & Hold', linewidth=2)
ax[0].plot(df.index, df['cum_strat'], label='Gini-Enhanced PHMC', linewidth=2)
ax[0].legend(fontsize=11)
ax[0].set_title('Cumulative Performance with Gini-Adjusted Position Sizing', fontsize=13, fontweight='bold')
ax[0].set_ylabel('Cumulative Return')
ax[0].grid(True, alpha=0.3)

ax[1].plot(df.index, df['Z'], label='Topo Z-score', color='navy', linewidth=1.5)
ax[1].plot(df.index, df['entry_threshold_adj'], label='Gini-Adjusted Entry Threshold', 
           color='red', linestyle='--', linewidth=1.5)
ax[1].axhline(params['exit_z'], linestyle=':', color='orange', label='Exit Threshold')
ax[1].set_ylabel('Z-Score')
ax[1].legend()
ax[1].grid(True, alpha=0.3)
ax[1].set_title('Z-Score with Gini-Adjusted Entry Thresholds')

ax[2].plot(df.index, df['gini_returns'], label='Market Returns Gini', color='green', linewidth=1.5)
ax[2].plot(df.index, df['gini_strategy'], label='Strategy Returns Gini', color='purple', linewidth=1.5)
ax[2].axhline(params['gini_threshold'], linestyle='--', color='red', 
              label=f"Reduction Threshold ({params['gini_threshold']})")
ax[2].set_ylabel('Gini Coefficient')
ax[2].set_xlabel('Date')
ax[2].legend()
ax[2].grid(True, alpha=0.3)
ax[2].set_title('Rolling Gini Coefficients (Return Concentration)')

plt.tight_layout()
plt.show()

# Plot 2: Lorenz Curves
plot_lorenz_curve(df['returns'], df['strat_ret'])
plt.show()

# Save results
display_df = df[['close', 'returns', 'atr', 'run_len_pos', 'topo_trend', 'Z', 
                 'gini_returns', 'gini_strategy', 'entry_threshold_adj', 
                 'entry', 'position', 'strat_ret', 'cum_strat']].copy()
display_df.to_csv('/mnt/data/s003_phmc_gini_backtest.csv', index=True)
print("\nData saved to: /mnt/data/s003_phmc_gini_backtest.csv")
print("\nFirst 30 rows:")
print(display_df.head(30))
