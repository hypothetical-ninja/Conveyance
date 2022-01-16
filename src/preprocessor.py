import numpy as np
import re
import os
import pandas as pd
import json
from pandas.api.types import is_string_dtype, is_numeric_dtype
import yaml
from sklearn.preprocessing import normalize

#arbitrary dropoff threshold decided after analysing the dropoff distances. 99th percentile value is 687m

class Preprocess:
    def __init__(self, dataframe_path, config):
        self.df = pd.read_csv(dataframe_path)
        self.config = config


    # only for numeric variables
    def remove_outliers(self, colnames):
        for col in colnames:
            if not is_numeric_dtype(self.df[col]):
                continue
            if col == 'logistics_dropoff_distance':
                self.df = self.df[self.df[col] <= self.config['thresholds']['dropoff_distance']]
            else:
                self.df = self.df[np.abs(self.df[col] - self.df[col].mean()) <= (3 * self.df[col].mad())]



    '''auto select features utility: identify numeric columns, categorical columns excluding target variable and other exceptions
    numeric columns: if not sufficient variance, drop
    categorical columns: if distinct values is greater than 10% of dataset or 100k (whichever is greater) then drop'''
    def auto_select_features(self, use_variance_for_cat=True):
        num_features= self.df.select_dtypes(include=['int64','float64']).drop(self.config.ml.target_column, axis=1).columns
        cat_features = self.df.select_dtypes(include=['object','bool']).drop([self.config.ml.target_column], axis=1).columns

        if use_variance_for_cat:
            num_features += cat_features
        #use variance to determine
        selected_features = []
        for feature in num_features:
            subset = self.df[feature, self.config['ml']['target_column']]
            variable_variance = normalize(subset).var()
            if variable_variance >= self.config['thresholds']['numeric_variance_threshold']:
                selected_features.append(feature)

        if not use_variance_for_cat:
            lower_thresh = self.config['thresholds']['categorical_threshold'] * len(self.df)
            upper_thresh = (1-self.config['thresholds']['categorical_threshold']) * len(self.df)
            for feature in cat_features:
                subset = self.df[feature]
                num_vals = len(pd.unqiue(self.df[feature]))
                if num_vals >= lower_thresh and num_vals <= upper_thresh:
                    selected_features.append(feature)

        selected_features.append(self.config['ml']['target_column'])
        self.df = self.df[selected_features]


    '''stratified sampling of a dataframe against a categorical column'''
    def stratified_sample(self, against, fraction=0.1):
        if self.df[against].dtype == np.int64 or self.df[against].dtype == np.float64:
            #throw some error or convert to categorical
            return None
        unique_values = pd.unique(self.df[against].values)
        sampled = pd.DataFrame()
        for value in unique_values:
            subset = self.df[self.df[against] == value]
            subset.sample(frac=fraction, inplace=True)
            sampled = sampled.append(subset)
        return sampled

    def extract_hour(self, field):
        return re.findall(r'T(.*?)\:', field, re.I)[0]

    #currently hard-coded
    def create_geohash_features(self):
        self.df['is_w21xz'] = self.df['delivery_geohash_precision8'].apply(lambda z: z.startswith('w21xz'))
        self.df['is_w23b7'] = self.df['delivery_geohash_precision8'].apply(lambda z: z.startswith('w23b7'))

    def add_features(self):
        self.df['hour'] = self.df['created_timestamp_local'].apply(self.extract_hour)
        self.df['delivery_geohash_precision5'] = self.df['delivery_geohash_precision8'].apply(lambda z: z[:5])

    def write_df(self, output_path):
        self.df.to_csv(output_path, index=False)


# if __name__ == "__main__":
