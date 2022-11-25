import pandas as pd

from datetime import datetime
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
import os

def etl(ti):

    data_paths = {}
    data = {}
    for dirname, _, filenames in os.walk("data/provided"):
        for filename in filenames:
            print(os.path.join(dirname, filename))

            data_paths[filename] = f"{dirname}/{filename}"

    for file, path in data_paths.items():
        data[file.split('.')[0]] = pd.read_csv(path)

    filter_no_neg_price = lambda df: (df.item_price > 0)
    filter_no_returns = lambda df: (df.item_cnt_day >= 0)
    filter_outliers__price = lambda df: (df.item_price < 178179)
    filter_outliers__cnt = lambda df: (df.item_cnt_day < 250)

    etl__filter_registry = {
        'sales_train': [
            filter_no_neg_price,
            filter_no_returns,
            filter_outliers__price,
            filter_outliers__cnt,
        ]
    }

    def etl__apply_filters(data):
        def _process_tbl(df, handlers):
            _df = df.copy()
            for handler in handlers:
                _df = _df[handler(_df)]

            return _df
        imm_data = {
            tbl_name: _process_tbl(
                df=data[tbl_name],
                handlers=etl__filter_registry[tbl_name] if tbl_name in etl__filter_registry.keys() else []) for tbl_name in
            data.keys()
        }
        return imm_data

    imm_data = etl__apply_filters(data)
    raw_item__df = imm_data['items']
    raw_item_cat__df = imm_data['item_categories']
    raw_sales__df = imm_data['sales_train']
    raw_shop__df = imm_data['shops']

    imm_flat_df = (raw_sales__df.merge(raw_item__df, on='item_id', how='left')\
                   .merge(raw_item_cat__df, on='item_category_id', how='left')\
                   .merge(raw_shop__df, on='shop_id', how='left'))
    imm_flat_df['date'] = pd.to_datetime(imm_flat_df["date"])
    imm_flat_df['date__days_since_hstart'] = (imm_flat_df['date'] - min(imm_flat_df['date'])).dt.days
    imm_flat_df['date__weeks_since_hstart'] = imm_flat_df['date__days_since_hstart'] // 7
    imm_flat_df['date__day_of_month'] = imm_flat_df['date'].dt.day
    imm_flat_df['date__day_of_week'] = imm_flat_df['date'].dt.dayofweek
    imm_flat_df['date__week_of_year'] = imm_flat_df['date'].dt.weekofyear
    imm_flat_df['date__month_of_year'] = imm_flat_df['date'].dt.month
    imm_flat_df['date__year'] = imm_flat_df['date'].dt.year
    imm_flat_df.rename({'date_block_num': 'date__month_since_hstart'}, inplace=True, axis=1)
    imm_flat_df.to_csv("data/export/work_df.csv")


with DAG("etl", schedule_interval="@once", start_date=datetime(2022, 1, 1, 1, 1)) as dag:
    etl_task = PythonOperator(task_id="etl", python_callable=etl)
    etl_task

