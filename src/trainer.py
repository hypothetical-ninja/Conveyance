import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.kernel_approximation import RBFSampler


class MLTrainer:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.all_data = pd.read_csv(self.data_path)
        self.config = config

    def shuffle_df(self, n=0.3, subset=False):
        if subset:
            self.all_data.sample(fraction=n, inplace=True)
        else:
            self.all_data.sample(fraction=1, inplace=True)


    def verify_columns(self):
        cols_to_keep = self.config['ml']['columns_to_keep'] + self.config['ml']['target_column']
        self.all_data = self.all_data[cols_to_keep]

    def clear_cache(self):
        self.all_data = pd.DataFrame()

    def split(self, validation=False):
        y = self.all_data[self.config['ml']['target_column']]
        X = self.all_data.drop([self.config['ml']['target_column']], axis=1)
        test_size = self.config['ml']['test_size']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=19)
        if validation:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=test_size, random_state=19)

    def lr_model(self):
        self.clf = LinearRegression()
        self.pipe = Pipeline([("extraction_pipe", self.pipe), ("lr", self.cls)])

    def ridge_model(self):
        ridge_params = self.config['ml']['models']['rr_params']
        self.clf = Ridge(alpha=ridge_params["alpha"], max_iter=ridge_params["max_iter"], solver=ridge_params["solver"])
        self.pipe = Pipeline([("extraction_pipe", self.pipe), ("ridge", self.clf)])

    def lasso_model(self):
        lasso_params = self.config['ml']['models']['lar_params']
        self.clf = Lasso(selection=lasso_params["selection"], warm_start=lasso_params['warm_start'],
                         positive=lasso_params['positive'])
        self.pipe = Pipeline([("extraction_pipe", self.pipe), ("lasso", self.clf)])

    def dt_model(self):
        dt_params = self.config['ml']['models']['dt_params']
        self.clf = tree.DecisionTreeRegressor(min_samples_split=dt_params["min_samples_split"],
                                              criterion=dt_params["criterion"], max_depth=dt_params["max_depth"])
        self.pipe = Pipeline([("extraction_pipe", self.pipe), ("decision_tree", self.clf)])


    def rf_model(self):
        rf_params = self.config['ml']['models']['rf_params']


    def svl(self):
        svl_params = self.config['ml']['models']['svl_params']
        self.cls = svm.SVR(kernel=svl_params["kernel"], max_iter=svl_params["max_iter"])
        self.pipe = Pipeline([("extraction_pipe", self.pipe), ("svm", self.cls)])


    def pipelinecreate(self, classifier):
        oe = OneHotEncoder()
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', oe)])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['ml']['numerical_columns']),
                ('cat', categorical_transformer, self.config['ml']['categorical_columns'])])
        self.pipe = Pipeline(steps=[('preprocessor', preprocessor),])


    def train(self):
        self.model = self.pipe.fit(self.X_train, self.y_train)

    def publish_results(self):
        rmse = mean_squared_error(self.y_pred, self.y_test, squared=False)
        mae = mean_absolute_error(self.y_pred,self.y_test)
        results = pd.DataFrame({'MAE':mae, 'RMSE':rmse, 'Model':self.pipe[-1]})
        results.to_csv('../results.csv', index=False)
        return None

    def test(self, model):
        self.y_pred = self.model.predict(self.X_test)

    def save_model(self, path):
        joblib.dump(self.model, path)
