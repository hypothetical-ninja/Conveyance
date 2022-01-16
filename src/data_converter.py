import pandas as pd
import numpy as np
import re
import os
import json


module = os.path.basename(__file__)
home_dir = os.path.expanduser("~")
package_dir = os.path.dirname(os.path.abspath(__file__))



def convert_to_dataframe(json_file):
    dfs_list = []
    for colname in json_file.keys():
        sub_dictionary = json_file[colname]
        subdf = pd.DataFrame.from_dict(sub_dictionary, orient='index', columns=[colname])
        dfs_list.append(subdf)
    df = pd.concat(dfs_list, axis=1).reset_index(level=0)

    return df


def read_raw_file(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            raw_json = json.load(f)
    else:
        raw_json = None
    return raw_json

def read_from_directory(path_to_folder):
    files = []
    for file in os.listdir(home_dir + path_to_folder):
        raw_json = read_raw_file(home_dir + path_to_folder + file)
        if raw_json is not None:
            files.append(raw_json)
    return files


if __name__=='__main__':
    json_file = read_raw_file('../../../raw_data/location_task_no_nulls.json')
    convert_to_dataframe(json_file)