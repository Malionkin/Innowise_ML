import pandas as pd
from datetime import datetime
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
from itertools import product


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
        # Splitting shop name to get the city name
        shops["city"] = shops["shop_name"].str.split(" ").map(lambda x: x[0])
        shops["city"] = shops["city"].replace("!", "", regex=True)

        # Label encoding so that it is more readable
        shops["city_code"] = LabelEncoder().fit_transform(shops["city"])
        shops.drop(["city"], axis=1, inplace=True)
        shops.drop(["shop_name"], axis=1, inplace=True)

        # Splitting category name to get the item type.
        cats["item_cat_split"] = cats["item_category_name"].str.split("-")
        cats["type"] = cats["item_cat_split"].map(lambda x: x[0].strip())
        cats["type_code"] = LabelEncoder().fit_transform(cats["type"])

        # Splitting category name to get the item type.
        cats["subcat"] = cats["item_cat_split"].map(
            lambda x: x[1].strip() if len(x) > 1 else x[0].strip()
        )
        cats["subcat_code"] = LabelEncoder().fit_transform(cats["subcat"])
        cats = cats[["item_category_id", "type_code", "subcat_code"]]

        # Dropping item names from items df as we discussed
        # Create the date the product was first sold as a feature
        items["first_sale_date"] = train.groupby("item_id").agg({"date_block_num": "min"})[
            "date_block_num"
        ]

        items.drop(["item_name"], axis=1, inplace=True)
        train = (
            train.merge(items, on="item_id", how="left")
            .merge(shops, on="shop_id", how="left")
            .merge(cats, on="item_category_id", how="left")
        )

        test = (
            test.merge(items, on="item_id", how="left")
            .merge(shops, on="shop_id", how="left")
            .merge(cats, on="item_category_id", how="left")
        )
        train_monthly = train.groupby(
            [
                "date_block_num",
                "shop_id",
                "item_category_id",
                "item_id",
                "city_code",
                "type_code",
                "subcat_code",
            ],
            as_index=False,
        )
        train_monthly = train_monthly.agg(
            {"item_price": "mean", "item_cnt_day": ["sum", "mean", "count"]}
        ).reset_index(drop="True")
        train_monthly.columns = [
            "date_block_num",
            "shop_id",
            "item_category_id",
            "item_id",
            "city_code",
            "type_code",
            "subcat_code",
            "item_price",
            "item_cnt",
            "mean_item_cnt",
            "transactions",
        ]

        test["item_cnt"] = 0
        test["date_block_num"] = 34
        cols = [
            "shop_id",
            "item_id",
            "date_block_num",
            "item_category_id",
            "city_code",
            "type_code",
            "subcat_code",
            "first_sale_date",
        ]
        train_monthly = pd.concat(
            [train_monthly, test], ignore_index=True, sort=False, keys=cols
        )

        train_monthly["year"] = train_monthly["date_block_num"].apply(
            lambda x: ((x // 12) + 2013)
        )
        train_monthly["month"] = train_monthly["date_block_num"].apply(
            lambda x: (x % 12))

        train_monthly["revenue"] = train_monthly["item_price"] * \
            train_monthly["item_cnt"]

        # Adding min and max price of the item
        item_price = (
            train_monthly.sort_values("date_block_num")
            .groupby(["item_id"], as_index=False)
            .agg({"item_price": [np.min, np.max]})
        )
        item_price.columns = ["item_id",
                              "hist_min_item_price", "hist_max_item_price"]
        train_monthly = pd.merge(
            train_monthly, item_price, on="item_id", how="left")

        # Adding average price difference w.r.t the current month with min and max historic price
        train_monthly["price_increase"] = (
            train_monthly["item_price"] - train_monthly["hist_min_item_price"]
        )
        train_monthly["price_decrease"] = (
            train_monthly["hist_max_item_price"] - train_monthly["item_price"]
        )

        # Adding first month of selling of a product
        train_monthly["first_selling_date_block"] = train_monthly.groupby("item_id")[
            "date_block_num"
        ].min()

        # Adding a boolean if a product is a newly launched product or not
        train_monthly["is_new_product"] = (
            train_monthly["first_selling_date_block"] == train_monthly["date_block_num"]
        )

        # Rolling window features
        # Min value
        def f_min(x): return x.rolling(window=3, min_periods=1).min()
        # Max value
        def f_max(x): return x.rolling(window=3, min_periods=1).max()
        # Mean value
        def f_mean(x): return x.rolling(window=3, min_periods=1).mean()
        # Standard deviation
        def f_std(x): return x.rolling(window=3, min_periods=1).std()

        func_list = [f_min, f_max, f_mean, f_std]
        func_name = ["min", "max", "mean", "std"]

        for i in range(len(func_list)):
            train_monthly[("item_cnt_%s" % func_name[i])] = (
                train_monthly.sort_values("date_block_num")
                .groupby(["shop_id", "item_category_id", "item_id"])["item_cnt"]
                .apply(func_list[i])
            )

        # Fill the empty std features with 0
        train_monthly["item_cnt_std"].fillna(0, inplace=True)

        # Creating lag_features function to create the lags which helps model to predict the future month sales.

        def lag_features(df, lags, col_list):
            for col_name in col_list:
                tmp = df[["date_block_num", "shop_id", "item_id", col_name]]
                for i in lags:
                    shifted = tmp.copy()
                    shifted.columns = [
                        "date_block_num",
                        "shop_id",
                        "item_id",
                        col_name + "_lag_" + str(i),
                    ]
                    shifted["date_block_num"] += i
                    df = pd.merge(
                        df, shifted, on=["date_block_num", "shop_id", "item_id"], how="left"
                    )
            return df

        # 1.=====Creating monthly, bi-monthly, quarterly, half yearly of target feature===
        train_monthly = lag_features(
            train_monthly, [1, 2, 3, 6], [
                "item_cnt", "mean_item_cnt", "transactions"]
        )

        # 2.====Creating recent lag features for price feature===
        train_monthly = lag_features(train_monthly, [1, 2, 3], ["item_price"])

        # 3.====Creating lag feature for category sales====
        grp = train_monthly.groupby(["date_block_num", "item_category_id"]).agg(
            {"item_cnt": ["mean"]}
        )
        grp.columns = ["date_cat_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "item_category_id"], how="left"
        )
        train_monthly["date_cat_avg_item_cnt"] = train_monthly["date_cat_avg_item_cnt"].astype(
            np.float16
        )

        # 4.=====Creating lag features for shop level=====
        grp = train_monthly.groupby(
            ["date_block_num", "shop_id"]).agg({"item_cnt": ["mean"]})
        grp.columns = ["date_shop_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "shop_id"], how="left"
        )
        train_monthly["date_shop_cnt"] = train_monthly["date_shop_cnt"].astype(
            np.float16)

        # 5.====Creating lag features for shop and item category level====
        grp = train_monthly.groupby(["date_block_num", "shop_id", "item_category_id"]).agg(
            {"item_cnt": ["mean"]}
        )
        grp.columns = ["date_shop_cat_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "shop_id", "item_category_id"], how="left"
        )
        train_monthly["date_shop_cat_avg_item_cnt"] = train_monthly[
            "date_shop_cat_avg_item_cnt"
        ].astype(np.float16)

        # 6.=====Creating lag features for shop and item type=====
        grp = train_monthly.groupby(["date_block_num", "shop_id", "type_code"]).agg(
            {"item_cnt": ["mean"]}
        )
        grp.columns = ["date_shop_type_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "shop_id", "type_code"], how="left"
        )
        train_monthly["date_shop_type_avg_item_cnt"] = train_monthly[
            "date_shop_type_avg_item_cnt"
        ].astype(np.float16)

        # 7.====Creating lag features for shop and item subcateory type====
        grp = train_monthly.groupby(["date_block_num", "shop_id", "subcat_code"]).agg(
            {"item_cnt": ["mean"]}
        )
        grp.columns = ["date_shop_subcat_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "shop_id", "subcat_code"], how="left"
        )
        train_monthly["date_shop_subcat_avg_item_cnt"] = train_monthly[
            "date_shop_subcat_avg_item_cnt"
        ].astype(np.float16)

        # 8.====Creating lag features at city level sales====
        grp = train_monthly.groupby(
            ["date_block_num", "city_code"]).agg({"item_cnt": ["mean"]})
        grp.columns = ["date_city_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "city_code"], how="left"
        )
        train_monthly["date_city_avg_item_cnt"] = train_monthly[
            "date_city_avg_item_cnt"
        ].astype(np.float16)

        # 9.====Creating lag features at city and item code level====
        grp = train_monthly.groupby(["date_block_num", "item_id", "city_code"]).agg(
            {"item_cnt": ["mean"]}
        )
        grp.columns = ["date_item_city_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "item_id", "city_code"], how="left"
        )
        train_monthly["date_item_city_avg_item_cnt"] = train_monthly[
            "date_item_city_avg_item_cnt"
        ].astype(np.float16)

        # 10.====Creating lag featuures at item type level====
        grp = train_monthly.groupby(
            ["date_block_num", "type_code"]).agg({"item_cnt": ["mean"]})
        grp.columns = ["date_type_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "type_code"], how="left"
        )
        train_monthly["date_type_avg_item_cnt"] = train_monthly[
            "date_type_avg_item_cnt"
        ].astype(np.float16)

        # 11.=====Creating lag features at item sub category level====
        grp = train_monthly.groupby(["date_block_num", "subcat_code"]).agg(
            {"item_cnt": ["mean"]}
        )
        grp.columns = ["date_subtype_avg_item_cnt"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "subcat_code"], how="left"
        )
        train_monthly["date_subtype_avg_item_cnt"] = train_monthly[
            "date_subtype_avg_item_cnt"
        ].astype(np.float16)

        # Applying lag function
        lag_on_list = [
            "date_cat_avg_item_cnt",
            "date_shop_cnt",
            "date_shop_cat_avg_item_cnt",
            "date_shop_type_avg_item_cnt",
            "date_shop_subcat_avg_item_cnt",
            "date_city_avg_item_cnt",
            "date_item_city_avg_item_cnt",
            "date_type_avg_item_cnt",
            "date_subtype_avg_item_cnt",
            "revenue",
        ]  # Added revenue lags as well

        train_monthly = lag_features(train_monthly, [1, 2, 3], lag_on_list)
        train_monthly.drop(lag_on_list, axis=1, inplace=True)

        # Creating last 3 month average of sales
        train_monthly["qmean"] = train_monthly[
            ["item_cnt_lag_1", "item_cnt_lag_2", "item_cnt_lag_3"]
        ].mean(skipna=True, axis=1)

        train_monthly["qmean_rev"] = train_monthly[
            ["revenue_lag_1", "revenue_lag_2", "revenue_lag_3"]
        ].mean(skipna=True, axis=1)

        # Getting some lag ratios to get some kind of trend
        train_monthly["item_cat_lag1_ratio"] = (
            train_monthly["item_cnt_lag_1"] / train_monthly["item_cnt_lag_2"]
        )
        train_monthly["item_cat_lag1_ratio"] = (
            train_monthly["item_cat_lag1_ratio"].replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
        )

        train_monthly["item_cat_lag2_ratio"] = (
            train_monthly["item_cnt_lag_2"] / train_monthly["item_cnt_lag_3"]
        )
        train_monthly["item_cat_lag2_ratio"] = (
            train_monthly["item_cat_lag2_ratio"].replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
        )

        # Adding revenue ratio to get additional trend features
        train_monthly["rev_lag1_ratio"] = (
            train_monthly["revenue_lag_1"] / train_monthly["revenue_lag_2"]
        )
        train_monthly["rev_lag2_ratio"] = (
            train_monthly["revenue_lag_2"] / train_monthly["revenue_lag_3"]
        )

        # Additional Trend features
        grp = train.groupby(["item_id"]).agg({"item_price": ["mean"]})
        grp.columns = ["avg_item_price"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(train_monthly, grp, on=[
                                 "item_id"], how="left")
        train_monthly["avg_item_price"] = train_monthly["avg_item_price"].astype(
            np.float16)

        grp = train.groupby(["date_block_num", "item_id"]
                            ).agg({"item_price": ["mean"]})
        grp.columns = ["date_avg_item_price"]
        grp.reset_index(inplace=True)

        train_monthly = pd.merge(
            train_monthly, grp, on=["date_block_num", "item_id"], how="left"
        )
        train_monthly["date_avg_item_price"] = train_monthly["date_avg_item_price"].astype(
            np.float16
        )

        # Taking 6 months of price lags to get the delta of price for trend variable
        lags = [1, 2, 3, 4, 5, 6]
        train_monthly = lag_features(
            train_monthly, lags, ["date_avg_item_price"])

        # Calculating the deviation from average price to get the delta at different lag
        for i in lags:
            train_monthly["delta_price_lag_" + str(i)] = (
                train_monthly["date_avg_item_price_lag_" + str(i)]
                - train_monthly["avg_item_price"]
            ) / train_monthly["avg_item_price"]

        # Logic to get the closest non null value to calculate the delta value

        def select_trend(row):
            for i in lags:
                if row["delta_price_lag_" + str(i)]:
                    return row["delta_price_lag_" + str(i)]
            return 0

        train_monthly["price_trend"] = train_monthly.apply(
            select_trend, axis=1)
        train_monthly["price_trend"] = train_monthly["price_trend"].astype(
            np.float16)
        train_monthly["price_trend"].fillna(0, inplace=True)

        drop_cols = ["avg_item_price", "date_avg_item_price"]
        for i in lags:
            drop_cols += ["date_avg_item_price_lag_" + str(i)]
            drop_cols += ["delta_price_lag_" + str(i)]

        train_monthly.drop(drop_cols, axis=1, inplace=True)

        # replaceing inf and -inf with 0
        train_monthly = train_monthly.replace(
            [np.inf, -np.inf], np.nan).fillna(0)

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
