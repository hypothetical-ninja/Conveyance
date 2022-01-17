import pandas as pd
import joblib
import preprocessor
import yaml

import warnings
warnings.filterwarnings('ignore')


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
            df = None
        return df

    def preprocess_data(self):
        prep = preprocessor.Preprocess(self.test_data, self.config, False)
        prep.remove_outliers(self.config['ml']['target_column'])
        if self.config['data-ingestion']['auto_select_features']:
            prep.auto_select_features(use_variance_for_cat=self.config['data-ingestion']['use_variance_for_categories'])

        prep.add_features()
        prep.write_df('../' + self.config['paths']['processed_inference'])

    def predict(self):
        df = pd.read_csv('../' + self.config['paths']['processed_inference'])
        df = self.verify_columns(df)
        if df is None:
            print("columns mismatch.")
            return None
        y_pred = self.model.predict(df)
        df['predicted_distance'] = y_pred
        return df



if __name__ == '__main__':
    with open('../config.yaml') as file:
        config = yaml.safe_load(file)['configuration']
    model_path = '../model/' + config['ml']['inference_model']
    test_data_path = config['paths']['test_data_path']

    tester = MLInference(model_path, test_data_path, config)
    tester.preprocess_data()
    outcome = tester.predict()
    outcome.to_csv(config['paths']['test_output_path'], index=False)
    print("Completed..!")


