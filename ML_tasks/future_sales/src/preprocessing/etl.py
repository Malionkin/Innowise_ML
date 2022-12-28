import pandas as pd
from datetime import datetime
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
from itertools import product
import tasks_for_transform as t


class ETL():
    """This class provides the extract-transform-load process
        Includes 3 methods (1 for each step of ETL-process)
    """

    MAX_PRICE = 40000
    MAX_CNT = 250

    def extract_data(self):
        """Extracting provided data from several files"""

        data_paths = {}
        data = {}
        for dirname, _, filenames in os.walk("./competitive-data-science-predict-future-sales/provided_data"):
            for filename in filenames:
                print(os.path.join(dirname, filename))
                data_paths[filename] = f"{dirname}/{filename}"
        for file, path in data_paths.items():
            data[file.split('.')[0]] = pd.read_csv(path)
        return data

    def clean_data(self, data):
        """transforms data according to written filters"""
        self.data = data
        def filter_no_neg_price(df): return (df.item_price > 0)
        #def filter_no_returns(df): return (df.item_cnt_day >= 0)
        def filter_outliers__price(df): return (df.item_price < MAX_PRICE)
        def filter_outliers__cnt(df): return (df.item_cnt_day < MAX_CNT)

        etl__filter_registry = {
            'sales_train': [
                filter_no_neg_price,
                # filter_no_returns,
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
        return imm_data

    def transform_data_for_feature_engineering(self):
        items = pd.read_csv(
            './competitive-data-science-predict-future-sales/transformed_data/items.csv')
        shops = pd.read_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/shops.csv")
        cats = pd.read_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/item_categories.csv")
        train = pd.read_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/sales_train.csv")
        # Creating a data dictionary for later usage
        train_data_dict = {"items": items,
                           "shops": shops,
                           "cats": cats,
                           "train": train}
        test = pd.read_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/test.csv").set_index("ID")
        shops = t.transform_shops(shops)
        cats = t.transform_cats(cats)
        items = t.transform_items(items, train)
        train = t.create_train(train, items, shops, cats)
        test = t.create_test(test, items, shops, cats)
        train_monthly = t.create_train_monthly(train, test)
        train_monthly = t.creating_lag_features(train_monthly)
        train_monthly = t.add_trends(train_monthly, train)
        train_monthly["item_cnt"] = train_monthly["item_cnt"].clip(0, 20)
        return train_monthly

    def load_for_eda(self, imm_data):
        """Loading data into one big dataframe for easier global analytics and several small files for convinient EDA"""

        self.imm_data = imm_data
        raw_item__df = imm_data['items']
        raw_item_cat__df = imm_data['item_categories']
        raw_sales__df = imm_data['sales_train']
        raw_shop__df = imm_data['shops']

        imm_flat_df = (raw_sales__df.merge(raw_item__df, on='item_id', how='left')
                       .merge(raw_item_cat__df, on='item_category_id', how='left')
                       .merge(raw_shop__df, on='shop_id', how='left'))
        imm_flat_df.to_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/work_df.csv", index=False)
        raw_item__df.to_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/items.csv", index=False)
        raw_item_cat__df.to_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/item_categories.csv", index=False)
        raw_sales__df.to_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/sales_train.csv", index=False)
        raw_shop__df.to_csv(
            "./competitive-data-science-predict-future-sales/transformed_data/shops.csv", index=False)

    def load_for_modelling(self, train_monthly):
        self.train_monthly = train_monthly
        train_monthly.to_pickle(
            "./competitive-data-science-predict-future-sales/transformed_data/train_monthly.pkl")
