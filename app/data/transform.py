import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

from app.data.fetch import final_fetch_company_data, final_fetch_broad_index_data, final_fetch_sector_index_data

#  ================================================ company ================================================ #

def transform_company_data() -> pd.DataFrame:
    df = final_fetch_company_data()

    df = df.sort_values(by=['Company', 'Date'], ascending=True).reset_index(drop=True)

    group = df.groupby('Company')

    # 1. Log Return
    df['log_return'] = group['Close'].transform(lambda x: np.log(x / x.shift(1)))

    # 2. Daily Return
    df['daily_return'] = (df['Close'] - df['Open']) / df['Open']

    # 3. High-Low Spread
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']

    # 4. Range ratio (VERY IMPORTANT)
    df['range_ratio'] = (df['High'] - df['Low']) / df['Low']

    # 5. Rolling Volatility
    df['volatility_5'] = group['log_return'].transform(lambda x: x.rolling(5).std())
    df['volatility_10'] = group['log_return'].transform(lambda x: x.rolling(10).std())

    # 6. Rolling Mean
    df['rolling_mean_5'] = group['log_return'].transform(lambda x: x.rolling(5).mean())
    df['rolling_mean_10'] = group['log_return'].transform(lambda x: x.rolling(10).mean())

    # 7. Volume Features
    df['volume_change'] = group['Volume'].pct_change()

    df['vol_mean_5'] = group['Volume'].transform(lambda x: x.rolling(5).mean())
    df['volume_spike'] = df['Volume'] / df['vol_mean_5']

    # 8. ATR
    df['prev_close'] = group['Close'].shift(1)

    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prev_close']).abs()
    df['tr3'] = (df['Low'] - df['prev_close']).abs()

    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    df['atr_4'] = group['true_range'].transform(lambda x: x.rolling(4).mean())
    df['atr_8'] = group['true_range'].transform(lambda x: x.rolling(8).mean())

    # 9. Lag Features
    df['vol_lag_1'] = group['volatility_5'].shift(1)
    df['vol_lag_2'] = group['volatility_5'].shift(2)
    df['vol_lag_3'] = group['volatility_5'].shift(3)

    # 10. Moving Averages
    df['sma_5'] = group['Close'].transform(lambda x: x.rolling(5).mean())
    df['sma_10'] = group['Close'].transform(lambda x: x.rolling(10).mean())

    df['price_sma_ratio'] = df['Close'] / df['sma_5']

    # 11. Time Features
    df['Date'] = pd.to_datetime(df['Date'])

    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month

    # Target: Next day volatility (%)
    df['next_high'] = group['High'].shift(-1)
    df['next_low'] = group['Low'].shift(-1)

    # Stable target 
    df['target_volatility'] = (
        (df['next_high'] - df['next_low']) / df['Close']
    ) * 100

    # Lag Features
    df['tr_vol_lag_1'] = group['true_range'].shift(1)
    df['tr_vol_lag_2'] = group['true_range'].shift(2)
    df['tr_vol_lag_3'] = group['true_range'].shift(3)
    df['tr_vol_lag_5'] = group['true_range'].shift(5)
    df['tr_vol_lag_10'] = group['true_range'].shift(10)

    df['tr_vol_lag_1_norm'] = df['tr_vol_lag_1'] / df['Close']
    
    # Rolling Features
    df['vol_roll_mean_5'] = group['true_range'].shift(1).transform(lambda x: x.rolling(5).mean())
    df['vol_roll_std_5'] = group['true_range'].shift(1).transform(lambda x: x.rolling(5).std())
    df['vol_ema_5'] = group['true_range'].shift(1).transform(lambda x: x.ewm(span=5).mean())
    df['vol_roll_mean_5_norm'] = df['vol_roll_mean_5'] / df['Close']

    # Normalize
    df['atr_4_norm'] = df['atr_4'] / df['Close']
    df['true_range_norm'] = df['true_range'] / df['Close']

    df['hl_log_sq'] = np.log(df['High'] / df['Low']) ** 2
    df['parkinson_vol'] = group['hl_log_sq'].transform(lambda x: x.rolling(5).mean())

    df['gk_vol'] = (
        0.5 * (np.log(df['High'] / df['Low']) ** 2)
        - (2*np.log(2)-1)*(np.log(df['Close']/df['Open'])**2)
    )

    df['gk_vol_5'] = group['gk_vol'].transform(lambda x: x.rolling(5).mean())

    df['vol_regime'] = df['volatility_10'] / (df['volatility_5'] + 1e-6)

    df['return_5'] = group['Close'].pct_change(5)
    df['return_10'] = group['Close'].pct_change(10)

    df['gap'] = (df['Open'] - df['prev_close']) / df['prev_close']

    df['vol_of_vol'] = group['volatility_5'].transform(lambda x: x.rolling(5).std())

    # Clean
    df = df.dropna().reset_index(drop=True)

    return df

#  ================================================ broad index ================================================ #

def transform_broad_index_data() -> pd.DataFrame:
    broad_index_df = final_fetch_broad_index_data()

    broad_index_df = broad_index_df.sort_values(by=['Index', 'Date'], ascending=True).reset_index(drop=True)

    # Log Return
    broad_index_df['idx_log_return'] = broad_index_df.groupby('Index')['Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Daily Return
    broad_index_df['idx_daily_return'] = (
        (broad_index_df['Close'] - broad_index_df['Open']) / broad_index_df['Open']
    )

    # High-Low Spread
    broad_index_df['idx_hl_spread'] = (
        (broad_index_df['High'] - broad_index_df['Low']) / broad_index_df['Close']
    )

    # 4. VOLATILITY
    broad_index_df['idx_volatility_4'] = broad_index_df.groupby('Index')['idx_log_return'].transform(
        lambda x: x.rolling(4).std()
    )
    broad_index_df['idx_volatility_8'] = broad_index_df.groupby('Index')['idx_log_return'].transform(
        lambda x: x.rolling(8).std()
    )

    # 5. TREND
    broad_index_df['idx_sma_5'] = broad_index_df.groupby('Index')['Close'].transform(
        lambda x: x.rolling(5).mean()
    )
    broad_index_df['idx_price_sma_ratio'] = (
        broad_index_df['Close'] / broad_index_df['idx_sma_5']
    )

    # 6. RANGE
    broad_index_df['idx_range_ratio'] = (
        (broad_index_df['High'] - broad_index_df['Low']) / broad_index_df['Low']
    )

    # 7. LAG FEATURES
    broad_index_df['idx_vol_lag_1'] = broad_index_df.groupby('Index')['idx_volatility_4'].shift(1)
    broad_index_df['idx_vol_lag_2'] = broad_index_df.groupby('Index')['idx_volatility_4'].shift(2)

    # 8. CLEAN
    broad_index_df = broad_index_df.dropna().reset_index(drop=True)

    return broad_index_df


#  ================================================ sector index ================================================ #


def transform_sector_index_data() -> pd.DataFrame:
    sector_index_df = final_fetch_sector_index_data()

    sector_index_df = sector_index_df.sort_values(by=['Index', 'Date'], ascending=True).reset_index(drop=True)

    # Log Return
    sector_index_df['idx_log_return'] = sector_index_df.groupby('Index')['Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Daily Return
    sector_index_df['idx_daily_return'] = (
        (sector_index_df['Close'] - sector_index_df['Open']) / sector_index_df['Open']
    )

    # High-Low Spread
    sector_index_df['idx_hl_spread'] = (
        (sector_index_df['High'] - sector_index_df['Low']) / sector_index_df['Close']
    )

    # 4. VOLATILITY
    sector_index_df['idx_volatility_4'] = sector_index_df.groupby('Index')['idx_log_return'].transform(
        lambda x: x.rolling(4).std()
    )
    sector_index_df['idx_volatility_8'] = sector_index_df.groupby('Index')['idx_log_return'].transform(
        lambda x: x.rolling(8).std()
    )

    # 5. TREND
    sector_index_df['idx_sma_5'] = sector_index_df.groupby('Index')['Close'].transform(
        lambda x: x.rolling(5).mean()
    )
    sector_index_df['idx_price_sma_ratio'] = (
        sector_index_df['Close'] / sector_index_df['idx_sma_5']
    )

    # 6. RANGE
    sector_index_df['idx_range_ratio'] = (
        (sector_index_df['High'] - sector_index_df['Low']) / sector_index_df['Low']
    )

    # 7. LAG FEATURES
    sector_index_df['idx_vol_lag_1'] = sector_index_df.groupby('Index')['idx_volatility_4'].shift(1)
    sector_index_df['idx_vol_lag_2'] = sector_index_df.groupby('Index')['idx_volatility_4'].shift(2)

    # 8. CLEAN
    sector_index_df = sector_index_df.dropna().reset_index(drop=True)

    return sector_index_df