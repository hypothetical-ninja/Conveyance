import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import svm
import joblib
import preprocessor



class MLInference:
    def __init__(self, model_path, test_data_path, config):
        self.config = config
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.test_data_path = test_data_path
        self.test_data = pd.read_csv(self.test_data_path)

    def verify_columns(self, df):
        cols_to_keep = self.config['ml']['columns_to_keep'] + self.config['ml']['target_column']
        try:
            df = df[cols_to_keep]
        except KeyError as e:
            print(e)
            return None

    def preprocess_data(self):
        prep = preprocessor.Preprocess(self.config['paths']['dataframe_path'], self.config['configuration'])
        prep.remove_outliers(self.config['ml']['target_column'])
        if self.config['data-ingestion']['preprocess']:
            prep.auto_select_features(use_variance_for_cat=self.config['data-ingestion']['use_variance_for_category'])

        prep.add_features()
        prep.write_df(self.config['paths']['processed_inference'])

    def predict(self):
        df = pd.read_csv(self.config['paths']['processed_inference'])
        if self.verify_columns(df) == None:
            return None


