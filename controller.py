import os
import yaml
import src.preprocessor as preprocessor
import src.trainer as trainer
import src.data_converter as data_converter

import warnings
warnings.filterwarnings('ignore')

module = os.path.basename(__file__)
home_dir = os.path.expanduser("~")
package_dir = os.path.dirname(os.path.abspath(__file__))

'''controller function to call all class methods. Steps can be decided accordingly and modified..'''
def main(config):
    if config['data-ingestion']['read_raw']:
        json_object = data_converter.read_raw_file(config['paths']['raw_file_path'])
        df = data_converter.convert_to_dataframe(json_object)
        df.to_csv(package_dir + config['paths']['dataframe_path'], index=False)

    if config['data-ingestion']['preprocess']:
        prep = preprocessor.Preprocess(package_dir + '/' + config['paths']['dataframe_path'], config)
        prep.remove_outliers(config['ml']['target_column'])
        if config['data-ingestion']['auto_select_features']:
            prep.auto_select_features(use_variance_for_cat=config['data-ingestion']['use_variance_for_categories'])

        prep.add_features()
        prep.write_df(package_dir + '/' + config['paths']['ml_input_path'])

    if config['ml']['training_mode']:
        trainer_obj = trainer.MLTrainer(package_dir + '/' + config['paths']['ml_input_path'], config)
        trainer_obj.shuffle_df()
        trainer_obj.verify_columns()
        trainer_obj.split(config['ml']['add_validation_set'])
        trainer_obj.pipelinecreate()
        trainer_obj.ridge_model()
        print("Fitting the model..")
        trainer_obj.train()
        print("Saving the model..")
        trainer_obj.save_model(package_dir + '/' + config['paths']['model_directory'] + config['paths']['rr_path'])
        print("Saved..")
        trainer_obj.test()
        print("Test complete..")
        trainer_obj.publish_results()



if __name__=="__main__":
    with open(package_dir+'/config.yaml') as file:
        config = yaml.safe_load(file)['configuration']
    main(config)


