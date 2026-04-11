import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

from app.data.transform import transform_company_data, transform_broad_index_data, transform_sector_index_data
from app.data.risk_transform import risk_transform_company_data, risk_transform_broad_index_data, risk_transform_sector_index_data
from app.data.fetch import company_mapping

def final_mapped_data() -> pd.DataFrame:
    company = transform_company_data()
    broad_index = transform_broad_index_data()
    sector_index = transform_sector_index_data()

    broad_index = broad_index.add_prefix('broad_')
    sector_index = sector_index.add_prefix('sector_')

    mapping_df = pd.DataFrame(company_mapping())

    company = pd.merge(company, mapping_df, left_on='Company', right_on='stock', how='left')

    data = pd.merge(
        company,
        broad_index,
        left_on=['Date', 'broad_index'],
        right_on=['broad_Date', 'broad_Index'],
        how='left'
    )

    data = pd.merge(
        data,
        sector_index,
        left_on=['Date', 'sectoral_index'],
        right_on=['sector_Date', 'sector_Index'],
        how='left'
    )

    data = data.drop(columns=['broad_Date', 'sector_Date'])

    #  Correlation
    data['rel_vol_market'] = data['volatility_5'] / (data['broad_idx_volatility_4'] + 1e-6)
    data['rel_vol_sector'] = data['volatility_5'] / (data['sector_idx_volatility_4'] + 1e-6)

    data['corr_market_5'] = (
        data.groupby('Company')
        .apply(lambda x: x['log_return'].rolling(5).corr(x['broad_idx_log_return']))
        .reset_index(level=0, drop=True)
    )
    
    data['corr_sector_5'] = (
        data.groupby('Company')
        .apply(lambda x: x['log_return'].rolling(5).corr(x['sector_idx_log_return']))
        .reset_index(level=0, drop=True)
    )

    data['market_beta_5'] = data.groupby('Company').apply(
        lambda x: (
            x['log_return'].rolling(5).cov(x['broad_idx_log_return']) /
            x['broad_idx_log_return'].rolling(5).var()
        )
    ).reset_index(level=0, drop=True)
    data['sector_beta_5'] = data.groupby('Company').apply(
        lambda x: (
            x['log_return'].rolling(5).cov(x['sector_idx_log_return']) /
            x['sector_idx_log_return'].rolling(5).var()
        )
    ).reset_index(level=0, drop=True)
    #  Correlation

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna().reset_index(drop=True)
    return data


def risk_final_mapped_data() -> pd.DataFrame:
    company = risk_transform_company_data()
    broad_index = risk_transform_broad_index_data()
    sector_index = risk_transform_sector_index_data()

    broad_index = broad_index.add_prefix('broad_')
    sector_index = sector_index.add_prefix('sector_')

    mapping_df = pd.DataFrame(company_mapping())

    company = pd.merge(company, mapping_df, left_on='Company', right_on='stock', how='left')

    data = pd.merge(
        company,
        broad_index,
        left_on=['Date', 'broad_index'],
        right_on=['broad_Date', 'broad_Index'],
        how='left'
    )

    data = pd.merge(
        data,
        sector_index,
        left_on=['Date', 'sectoral_index'],
        right_on=['sector_Date', 'sector_Index'],
        how='left'
    )

    data = data.drop(columns=['broad_Date', 'sector_Date'])

    #  Correlation
    data['rel_vol_market'] = data['volatility_5'] / (data['broad_idx_volatility_4'] + 1e-6)
    data['rel_vol_sector'] = data['volatility_5'] / (data['sector_idx_volatility_4'] + 1e-6)

    data['corr_market_5'] = (
        data.groupby('Company')
        .apply(lambda x: x['log_return'].rolling(5).corr(x['broad_idx_log_return']))
        .reset_index(level=0, drop=True)
    )
    
    data['corr_sector_5'] = (
        data.groupby('Company')
        .apply(lambda x: x['log_return'].rolling(5).corr(x['sector_idx_log_return']))
        .reset_index(level=0, drop=True)
    )

    data['market_beta_5'] = data.groupby('Company').apply(
        lambda x: (
            x['log_return'].rolling(5).cov(x['broad_idx_log_return']) /
            x['broad_idx_log_return'].rolling(5).var()
        )
    ).reset_index(level=0, drop=True)
    data['sector_beta_5'] = data.groupby('Company').apply(
        lambda x: (
            x['log_return'].rolling(5).cov(x['sector_idx_log_return']) /
            x['sector_idx_log_return'].rolling(5).var()
        )
    ).reset_index(level=0, drop=True)
    #  Correlation

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna().reset_index(drop=True)
    return data