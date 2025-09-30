import numpy as np
import pandas as pd

def gini_coefficient(x):
    """Calculate Gini coefficient (0=equal, 1=concentrated)"""
    x = np.abs(np.array(x))
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

def atr(high, low, close, n=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=1).mean()

def compute_run_lengths(returns):
    """Count consecutive positive return days"""
    signs = (returns > 0).astype(int)
    rl = np.zeros(len(signs), dtype=int)
    run_length = 0
    for i, s in enumerate(signs):
        run_length = run_length + 1 if s == 1 else 0
        rl[i] = run_length
    return pd.Series(rl, index=returns.index)

def topo_trend_from_runs(run_lengths, lookback=120, min_run_length=3):
    """Sum of significant positive runs in lookback window"""
    idx = run_lengths.index
    out = pd.Series(0.0, index=idx)
    rl = run_lengths.values
    n = len(rl)
    
    # Find run endpoints
    run_ends = []
    for i in range(n):
        if rl[i] > 0 and (i == n-1 or rl[i+1] == 0):
            run_ends.append((i, rl[i]))
    
    run_ends_idx = np.array([e[0] for e in run_ends])
    run_ends_len = np.array([e[1] for e in run_ends])
    
    # For each bar, sum qualifying runs in lookback
    for t in range(n):
        left = max(0, t - lookback + 1)
        mask = (run_ends_idx >= left) & (run_ends_idx <= t)
        if np.any(mask):
            lengths = run_ends_len[mask]
            lengths = lengths[lengths >= min_run_length]
            out.iloc[t] = lengths.sum() if len(lengths) > 0 else 0.0
    return out

class GiniPHMCStrategy:
    """
    Gini-Enhanced Positive Historical Momentum Continuation Strategy
    
    Entry: When trend Z-score exceeds Gini-adjusted threshold
    Exit: When Z-score drops below exit threshold
    Position Sizing: Scaled by Z-score, reduced in high-Gini periods
    """
    
    def __init__(self, params=None):
        self.params = params or {
            'atr_period': 14,
            'lookback': 120,
            'min_run_len': 3,
            'z_window': 100,
            'entry_z': 1.5,
            'exit_z': 0.7,
            'size_scale': 0.5,
            'max_pos': 2.0,
            'gini_window': 60,
            'gini_threshold': 0.65,
            'gini_scale': 0.5,
            'gini_adjustment': 0.3
        }
    
    def calculate_indicators(self, df):
        """Calculate all strategy indicators"""
        p = self.params
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        
        # Returns and ATR
        df['returns'] = close.pct_change().fillna(0)
        df['atr'] = atr(high, low, close, n=p['atr_period']).replace(0, np.nan).fillna(method='bfill')
        
        # Topological trend from run lengths
        df['run_len_pos'] = compute_run_lengths(df['returns'])
        df['topo_raw'] = topo_trend_from_runs(df['run_len_pos'], 
                                               lookback=p['lookback'], 
                                               min_run_length=p['min_run_len'])
        df['topo_trend'] = df['topo_raw'] / df['atr']
        
        # Z-score normalization
        df['topo_mu'] = df['topo_trend'].rolling(window=p['z_window'], min_periods=5).mean()
        df['topo_sigma'] = df['topo_trend'].rolling(window=p['z_window'], min_periods=5).std()
        df['topo_sigma'] = df['topo_sigma'].replace(0, np.nan).fillna(method='bfill')
        df['Z'] = (df['topo_trend'] - df['topo_mu']) / df['topo_sigma']
        
        # Gini coefficient on returns
        df['gini'] = df['returns'].rolling(window=p['gini_window'], min_periods=10).apply(
            lambda x: gini_coefficient(x), raw=True
        ).fillna(method='bfill').fillna(0.5)
        
        return df
    
    def generate_signals(self, df):
        """Generate entry/exit signals with Gini adjustment"""
        p = self.params
        
        # Gini-adjusted entry threshold
        df['entry_threshold'] = p['entry_z'] * (1 + p['gini_adjustment'] * df['gini'])
        df['signal'] = np.where(df['Z'] > df['entry_threshold'], 1,
                                np.where(df['Z'] < p['exit_z'], -1, 0))
        
        return df
    
    def calculate_positions(self, df):
        """Calculate position sizes with Gini scaling"""
        p = self.params
        
        # Base position from Z-score
        raw_pos = p['size_scale'] * df['Z'] / df['atr']
        df['position_base'] = raw_pos.clip(lower=0, upper=p['max_pos'])
        
        # Apply Gini penalty
        gini_penalty = np.where(
            df['gini'] > p['gini_threshold'],
            1 - p['gini_scale'] * ((df['gini'] - p['gini_threshold']) / (1 - p['gini_threshold'])),
            1.0
        )
        gini_penalty = np.clip(gini_penalty, 0.1, 1.0)
        
        df['position'] = df['position_base'] * gini_penalty
        
        # Apply signals
        df['position'] = np.where(df['signal'] == 1, df['position'], np.nan)
        df['position'] = df['position'].ffill().fillna(0)
        df['position'] = np.where(df['signal'] == -1, 0, df['position'])
        
        return df
    
    def run(self, price_df):
        """
        Execute strategy on price data
        
        Args:
            price_df: DataFrame with 'close' (required), 'high', 'low' (optional)
        
        Returns:
            DataFrame with signals, positions, and returns
        """
        df = price_df.copy()
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Calculate positions
        df = self.calculate_positions(df)
        
        # Calculate returns
        df['strategy_return'] = df['position'].shift(1) * df['returns']
        df['strategy_return'] = df['strategy_return'].fillna(0)
        df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
        df['cum_market'] = (1 + df['returns']).cumprod()
        
        return df
    
    def get_trade_log(self, df):
        """Extract trade entry/exit points"""
        pos_changes = df['position'].diff()
        entries = df[pos_changes > 0].copy()
        exits = df[pos_changes < 0].copy()
        
        trades = []
        for i, entry in entries.iterrows():
            exit_row = exits[exits.index > i].iloc[0] if len(exits[exits.index > i]) > 0 else None
            trades.append({
                'entry_date': i,
                'entry_price': entry['close'],
                'entry_z': entry['Z'],
                'entry_gini': entry['gini'],
                'position_size': entry['position'],
                'exit_date': exit_row.name if exit_row is not None else None,
                'exit_price': exit_row['close'] if exit_row is not None else None,
                'exit_z': exit_row['Z'] if exit_row is not None else None
            })
        
        return pd.DataFrame(trades)


# USAGE EXAMPLE
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    returns = np.random.normal(0.0002, 0.01, 1000)
    
    # Add trend periods
    returns[200:280] += 0.0015
    returns[600:720] += 0.0012
    
    price = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    df = pd.DataFrame({'close': price})
    df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.002, len(df))))
    df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.002, len(df))))
    
    # Run strategy
    strategy = GiniPHMCStrategy()
    results = strategy.run(df)
    
    # Performance stats
    total_return = results['cum_strategy'].iloc[-1] - 1
    market_return = results['cum_market'].iloc[-1] - 1
    trades = strategy.get_trade_log(results)
    
    print(f"Strategy Return: {total_return:.2%}")
    print(f"Market Return: {market_return:.2%}")
    print(f"Number of Trades: {len(trades)}")
    print(f"Avg Gini at Entry: {trades['entry_gini'].mean():.3f}")
    
    # Show last 10 rows
    print("\nLast 10 bars:")
    print(results[['close', 'Z', 'gini', 'signal', 'position', 'strategy_return']].tail(10))
