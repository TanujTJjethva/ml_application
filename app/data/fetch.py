import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

def company_mapping():
    return [
                { "stock": "RELIANCE.NS", "broad_index": "NIFTY_50", "sectoral_index": "NIFTY_METAL" },
                { "stock": "TCS.NS", "broad_index": "NIFTY_50", "sectoral_index": "NIFTY_IT" },
                { "stock": "HDFCBANK.NS", "broad_index": "NIFTY_50", "sectoral_index": "NIFTY_BANK" },
                { "stock": "INFY.NS", "broad_index": "NIFTY_100", "sectoral_index": "NIFTY_IT" },
                { "stock": "ICICIBANK.NS", "broad_index": "NIFTY_100", "sectoral_index": "NIFTY_BANK" },
                { "stock": "LT.NS", "broad_index": "NIFTY_100", "sectoral_index": "NIFTY_AUTO" },
                { "stock": "ITC.NS", "broad_index": "NIFTY_200", "sectoral_index": "NIFTY_FMCG" },
                { "stock": "HINDUNILVR.NS", "broad_index": "NIFTY_200", "sectoral_index": "NIFTY_FMCG" },
                { "stock": "KOTAKBANK.NS", "broad_index": "NIFTY_200", "sectoral_index": "NIFTY_BANK" },
                { "stock": "SBIN.NS", "broad_index": "NIFTY_200", "sectoral_index": "NIFTY_BANK" },
                { "stock": "BHARTIARTL.NS", "broad_index": "NIFTY_500", "sectoral_index": "NIFTY_AUTO" },
                { "stock": "ASIANPAINT.NS", "broad_index": "NIFTY_500", "sectoral_index": "NIFTY_FMCG" },
                { "stock": "MARUTI.NS", "broad_index": "NIFTY_500", "sectoral_index": "NIFTY_AUTO" },
                { "stock": "M&M.NS", "broad_index": "NIFTY_500", "sectoral_index": "NIFTY_AUTO" },
                { "stock": "SUNPHARMA.NS", "broad_index": "NIFTY_MIDCAP_100", "sectoral_index": "NIFTY_PHARMA" },
                { "stock": "DRREDDY.NS", "broad_index": "NIFTY_MIDCAP_100", "sectoral_index": "NIFTY_PHARMA" },
                { "stock": "CIPLA.NS", "broad_index": "NIFTY_MIDCAP_100", "sectoral_index": "NIFTY_PHARMA" },
                { "stock": "DIVISLAB.NS", "broad_index": "NIFTY_MIDCAP_100", "sectoral_index": "NIFTY_PHARMA" },
                { "stock": "AUROPHARMA.NS", "broad_index": "NIFTY_MIDCAP_100", "sectoral_index": "NIFTY_PHARMA" },
                { "stock": "JSWSTEEL.NS", "broad_index": "NIFTY_SMALLCAP_100", "sectoral_index": "NIFTY_METAL" },
                { "stock": "TATASTEEL.NS", "broad_index": "NIFTY_SMALLCAP_100", "sectoral_index": "NIFTY_METAL" },
                { "stock": "HINDALCO.NS", "broad_index": "NIFTY_SMALLCAP_100", "sectoral_index": "NIFTY_METAL" },
                { "stock": "VEDL.NS", "broad_index": "NIFTY_SMALLCAP_100", "sectoral_index": "NIFTY_METAL" },
                { "stock": "NMDC.NS", "broad_index": "NIFTY_SMALLCAP_100", "sectoral_index": "NIFTY_METAL" },
                { "stock": "WIPRO.NS", "broad_index": "NIFTY_50", "sectoral_index": "NIFTY_IT" },
                { "stock": "HCLTECH.NS", "broad_index": "NIFTY_100", "sectoral_index": "NIFTY_IT" },
                { "stock": "LTIM.NS", "broad_index": "NIFTY_200", "sectoral_index": "NIFTY_IT" },
                { "stock": "BAJAJ-AUTO.NS", "broad_index": "NIFTY_500", "sectoral_index": "NIFTY_AUTO" },
                { "stock": "HEROMOTOCO.NS", "broad_index": "NIFTY_500", "sectoral_index": "NIFTY_AUTO" }
        ]

def final_fetch_company_data() -> pd.DataFrame:
    stock_list = [item['stock'] for item in company_mapping()]

    tickers = yf.Tickers(stock_list)
    data = tickers.history(start='2026-02-01', end='2026-04-01')

    data = data.stack(level=1)
    data = data.reset_index()

    data = data.drop(columns=['Dividends', 'Stock Splits'])

    data = data.rename(columns={'Ticker':'Company'})

    return data

def final_fetch_broad_index_data() -> pd.DataFrame:
    csv_1 = pd.read_csv(BASE_DIR / 'app/data_source/broad_index/Nifty50HistoricalData.csv')
    csv_2 = pd.read_csv(BASE_DIR / 'app/data_source/broad_index/Nifty100HistoricalData.csv')
    csv_3 = pd.read_csv(BASE_DIR / 'app/data_source/broad_index/Nifty200HistoricalData.csv')
    csv_4 = pd.read_csv(BASE_DIR / 'app/data_source/broad_index/Nifty500HistoricalData.csv')
    csv_5 = pd.read_csv(BASE_DIR / 'app/data_source/broad_index/NIFTYMidcap100HistoricalData.csv')
    csv_6 = pd.read_csv(BASE_DIR / 'app/data_source/broad_index/NIFTYSmallcap100HistoricalData.csv')

    csv_1['Index'] = 'NIFTY_50'
    csv_2['Index'] = 'NIFTY_100'
    csv_3['Index'] = 'NIFTY_200'
    csv_4['Index'] = 'NIFTY_500'
    csv_5['Index'] = 'NIFTY_MIDCAP_100'
    csv_6['Index'] = 'NIFTY_SMALLCAP_100'

    data = pd.concat([csv_1, csv_2, csv_3, csv_4, csv_5, csv_6], ignore_index=True)

    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    
    data = data.drop(columns=['Vol.', 'Change %'])

    data = data.rename(columns={
                            'Price':'Close', 
                            })

    cols = [
        'Close',
        'Open',
        'High',
        'Low',
    ]

    data[cols] = (
        data[cols]
        .replace(['NaN', 'NAN', '', ' ', 'nan'], np.nan)
        .replace(',', '', regex=True)
        .astype(float)
    )

    data = data.dropna()

    return data

def final_fetch_sector_index_data() -> pd.DataFrame:
    csv_1 = pd.read_csv(BASE_DIR / 'app/data_source/sectoral_index/NiftyAutoHistoricalData.csv')
    csv_2 = pd.read_csv(BASE_DIR / 'app/data_source/sectoral_index/NiftyBankHistoricalData.csv')
    csv_3 = pd.read_csv(BASE_DIR / 'app/data_source/sectoral_index/NiftyFMCGHistoricalData.csv')
    csv_4 = pd.read_csv(BASE_DIR / 'app/data_source/sectoral_index/NiftyITHistoricalData.csv')
    csv_5 = pd.read_csv(BASE_DIR / 'app/data_source/sectoral_index/NiftyMetalHistoricalData.csv')
    csv_6 = pd.read_csv(BASE_DIR / 'app/data_source/sectoral_index/NiftyPharmaHistoricalData.csv')

    csv_1['Index'] = 'NIFTY_AUTO'
    csv_2['Index'] = 'NIFTY_BANK'
    csv_3['Index'] = 'NIFTY_FMCG'
    csv_4['Index'] = 'NIFTY_IT'
    csv_5['Index'] = 'NIFTY_METAL'
    csv_6['Index'] = 'NIFTY_PHARMA'

    data = pd.concat([csv_1, csv_2, csv_3, csv_4, csv_5, csv_6], ignore_index=True)

    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

    data = data.drop(columns=['Vol.', 'Change %'])

    data = data.rename(columns={
                        'Price':'Close', 
                    })

    cols = [
        'Close',
        'Open',
        'High',
        'Low',
    ]

    data[cols] = (
        data[cols]
        .replace(['NaN', 'NAN', '', ' ', 'nan'], np.nan)
        .replace(',', '', regex=True)
        .astype(float)
    )

    data = data.dropna()

    return data