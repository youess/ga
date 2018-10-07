# coding: utf-8

import os 
import gc 
import numpy as np 
import pandas as pd 
import json 
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings


warnings.simplefilter('ignore')

# https://www.kaggle.com/xavierbourretsicotte/localizing-utc-time-eda-and-walkthrough

def load_data(path, nrows=None):

    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(path, dtype={
        'fullVisitorId': 'str', 'visitId': 'str'}, 
        nrows=nrows,
        converters={column: json.loads for column in json_cols},
        engine="c")

    for col in json_cols:
        col_as_df = json_normalize(df[col])
        col_as_df.columns = [f'{col}.{subcolumn}' 
            for subcolumn in col_as_df.columns]
        df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)
    
    print(f"Loaded {os.path.basename(path)}, data shape: {df.shape}")
    return df 


def tidy_data(df):

    # Convert target into log scale
    y_name = 'totals.transactionRevenue'
    df[y_name] = df[y_name].astype(float)
    df[y_name+'_log'] = np.log1p(df[y_name])

    # Remove columns that has const value. May be indicate blank value
    na_vals = ['unknown.unknown', '(not set)', 'not available in demo dataset', 
        '(not provided)', '(none)', '<NA>']
    for c in df.columns:
        is_na = df[c].isin(na_vals)
        df.loc[is_na, c] = np.nan
    const_cols = [c for c in df.columns if df[c].notnull().sum() == 0]
    print(f"Only NA value column list: {const_cols}")
    df.drop(const_cols, axis=1, inplace=True)

    # Drop duplicate meaning of columns
    # Todo: date column could be add to test
    sc = ['date', 'sessionId', 'socialEngagementType']
    print(f"No use columns: {sc}")
    df.drop(sc, axis=1, inplace=True)

    # Convert to should-be-like dtype
    df['visitStartTime'] = pd.to_datetime(
        df['visitStartTime'].astype(int).astype(str), unit='s')

    df['visitId'] = df['visitId'].astype("int64")
    
    # device section
    df['device.isMobile'] = df['device.isMobile'].astype("int8")

    # totals section
    sc = [c for c in df.columns if c.startswith('totals.')]
    for c in sc:
        df[c] = df[c].astype(float)

    # trafficSource section
    # sc = [c for c in df.columns if c.startswith("trafficSource")]

    return df


def add_features(df):
    
    # Before do feature engineering, lets clean the data
    df = tidy_data(df)
    
    # Then we can process next part

    # Map the utc to local time
    ## day, hour, timestamp, day of week, month
    
    # fullVisitorId as some user level features

    # device
    c = "device.browser"
    prop = df[c].value_counts() / df.shape[0]
    other_browser = prop[prop < 0.01].index.tolist()
    idx = df[c].isin(other_browser)
    df.loc[idx, c] = 'other-browser'


    # geoNetwork

    # totals

    # trafficSource


    return df


def do_lgb(train, valid):
    model = None
    return model 


def write_sub(df, name=None):
    if name is None:
        name = 'submission.csv'
    df.to_csv(name, index=False)


def main(DBG=False):
    np.random.seed(123)
    nrows = 10000 if DBG else None

    train = load_data('../input/train.csv', nrows)
    sc = ['fullVisitorId', 'visitStartTime']
    train = train.sort_values(sc).reset_index(drop=True)
    test = load_data('../input/test.csv', nrows)
    test = test.sort_values(sc).reset_index(drop=True)

    y_name = 'totals.transactionRevenue'
    test[y_name] = 0
    tr_n = train.shape[0]
    train = train[test.columns.tolist()]
    train = train.append(test, ignore_index=True)
    del test 
    gc.collect()

    # revert data into train and test
    train = add_features(train)
    test = train.iloc[tr_n:, :].reset_index(drop=True)
    train = train.iloc[:tr_n, :].reset_index(drop=True)

    # split the train data into train and valid to tune model
    x_feat = train.columns.tolist()[:-1]
    train, valid = train_test_split(train[x_feat], train[y_name], test_size=0.2, random_state=2018)
    model = do_lgb(train, valid)

    # predict the test dataset to obtain target predicted value
    y_hat = model.predict(test)

    # write the result into submisssion file.
    write_sub(y_hat)


if __name__ == "__main__":
    main(True)