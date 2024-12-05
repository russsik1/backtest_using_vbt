import os, sys, traceback, io, time, zipfile, sqlite3
from typing import List, Dict
import datetime as dt
import pandas as pd
import numpy as np
import requests
# from tqdm import tqdm
from tqdm.notebook import tqdm








class Dataloader:


    def __init__(self):
        self.url = 'https://data.binance.vision/data/futures/um/monthly/klines/%s/1m/%s-1m-%s.zip'
        self.tfs: Dict[str, int] = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '8h': 480, '12h': 720, '1d': 1440}
        self.columns: List[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    def download_month(self, symbol: str, date: dt.datetime, pbar: tqdm=None) -> None:
        url = self.url%(symbol, symbol, date.strftime("%Y-%m"))
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(f'{z.namelist()[-1]}'), header=None)
        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        if df['open_time'][0] == 'open_time': df = df.drop(0, axis=0).reset_index(drop=True)
        df = df.rename(columns={'open_time': 'datetime', 'quote_volume': 'volume', 'volume': 'no metter'})
        df['datetime'] = df['datetime'].astype(np.int64)# + 60000
        if not os.path.exists(f'klines/{symbol}'):
            os.makedirs(f'klines/{symbol}')
        df[self.columns].to_csv(f'klines/{symbol}/{symbol} {date.strftime("%m.%Y")}.csv', index=False)
        if pbar:
            pbar.update(1)


    def download(self, symbol: str, start_date: str, end_date: str) -> None:
        start_date = dt.datetime.strptime(start_date, "%m.%Y")
        end_date = dt.datetime.strptime(end_date, "%m.%Y")
        month_count = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        with tqdm(total=month_count) as pbar:
            pbar.set_description(f'Cкачивание: {symbol}')
            for delta_date in pd.date_range(start_date, end_date, freq='MS'):
                self.download_month(symbol, delta_date, pbar)


    def read(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=self.columns)
        start_date = dt.datetime.strptime(start_date, "%m.%Y")
        end_date = dt.datetime.strptime(end_date, "%m.%Y")
        for delta_date in pd.date_range(start_date, end_date, freq='MS'):
            filename = f'klines/{symbol}/{symbol} {delta_date.strftime("%m.%Y")}.csv'
            df_monthly = pd.read_csv(filename)
            df = pd.concat([df, df_monthly])
        return df.sort_values(by='datetime', ascending=True)

    
    def convert_tf(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        df = df.copy()
        df['i'] = df['datetime']//(self.tfs[tf]*60000)*(self.tfs[tf]*60000)
        agg_functions = {'datetime': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        df = df.groupby('i').agg(agg_functions).reset_index()
        return df.drop(columns=['i'])




    def change_tf(self, basedf, tf, basetf='1m', extra_columns=[]):
        if tf=='1m' or len(basedf)==0 or tf==basetf: return basedf.copy()
        nontrasform_columns = [c for c in basedf.columns if c in ['close', 'datetime']] + extra_columns
        groupschema = {} #{'high': 'cummax', 'low': 'cummin', 'volume': 'cumsum'}
        if 'high' in basedf.columns: groupschema['high'] = 'cummax'
        if 'low' in basedf.columns: groupschema['low'] = 'cummin'
        if 'volume' in basedf.columns: groupschema['volume'] = 'cumsum'
        # [groupschema.pop(c) for c in groupschema.keys() if c not in basedf.columns]
        basedf = basedf.reset_index(drop=True).copy()
        basedf['s'] = basedf.index//int(self.tfs[tf]/self.tfs[basetf])
        df = basedf[nontrasform_columns].copy()
        if 'open' in basedf.columns: df = pd.concat([df, basedf.groupby('s')['open'].transform('first')], axis=1)
        if len(list(groupschema.keys()))>0: df = pd.concat([df, basedf.groupby(['s']).agg(groupschema)], axis=1)
        df = df.loc[(df.index+1)%(self.tfs[tf]/self.tfs[basetf])==0].reset_index(drop=True)
        df['datetime'] = df['datetime'] - self.tfs[basetf]*(int(self.tfs[tf]/self.tfs[basetf])-1)*60000
        return df.copy()

