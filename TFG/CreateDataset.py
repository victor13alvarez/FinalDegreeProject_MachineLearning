import json
import pprint
import pandas as pd


def read_file():
    file = open('matchlist_euw1.json')
    data = json.load(file)
    # data = file_intoDfTable(data)
    file.close()
    return data


def file_intoDfTable(data):
    dataset = []
    for match_id in data:
        dataset_row = {"match_id": match_id}
        dataset.append(dataset_row)
    df = pd.DataFrame(dataset)
    return df
