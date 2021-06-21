# imports
import os
import pandas as pd
from os import listdir
from os.path import isfile, join


def prepareDataSetForNeuralModel():
    if os.path.isfile('PreparedData.csv'):
        df = pd.read_csv('PreparedData.csv')
        return df

    MY_PATH = "Data/"
    DATAFRAME_PREPARED = "PreparedData.csv"
    DataFrame = []

    DatasetFromRIOT = [f for f in listdir(MY_PATH) if isfile(join(MY_PATH, f))]

    for datafile in DatasetFromRIOT:
        table_column = {}
        df = pd.read_csv(MY_PATH + datafile)
        mmr_list = df['CalculatedMMR'].values.tolist()
        if 'Unknown' in mmr_list:
            continue
        hotstreak_list = df['hot_streak'].values.tolist()
        duration_list = df['GameDuration'].values.tolist()
        duration = duration_list[0]
        for index, item in enumerate(hotstreak_list):
            if item is True:
                hotstreak_list[index] = 1
            else:
                hotstreak_list[index] = 0
        table_column['MMR player 0'] = mmr_list[0]
        table_column['MMR player 1'] = mmr_list[1]
        table_column['MMR player 2'] = mmr_list[2]
        table_column['MMR player 3'] = mmr_list[3]
        table_column['MMR player 4'] = mmr_list[4]
        table_column['MMR player 5'] = mmr_list[5]
        table_column['MMR player 6'] = mmr_list[6]
        table_column['MMR player 7'] = mmr_list[7]
        table_column['MMR player 8'] = mmr_list[8]
        table_column['MMR player 9'] = mmr_list[9]
        table_column['streak player 0'] = hotstreak_list[0]
        table_column['streak player 1'] = hotstreak_list[1]
        table_column['streak player 2'] = hotstreak_list[2]
        table_column['streak player 3'] = hotstreak_list[3]
        table_column['streak player 4'] = hotstreak_list[4]
        table_column['streak player 5'] = hotstreak_list[5]
        table_column['streak player 6'] = hotstreak_list[6]
        table_column['streak player 7'] = hotstreak_list[7]
        table_column['streak player 8'] = hotstreak_list[8]
        table_column['streak player 9'] = hotstreak_list[9]
        table_column['DURATION'] = duration
        DataFrame.append(table_column)

    dataframeToModel = pd.DataFrame(DataFrame)
    dataframeToModel.to_csv(DATAFRAME_PREPARED,index=False)
    return dataframeToModel