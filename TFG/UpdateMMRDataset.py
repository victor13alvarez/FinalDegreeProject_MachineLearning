import pandas as pd
from RiotAPI import calculate_player_MMR

from os import listdir
from os.path import isfile, join

MY_PATH = "Data/"


def split_elo(elo):
    return elo.translate({ord(i): None for i in '() \''}).split(",")


Datasets = [f for f in listdir(MY_PATH) if isfile(join(MY_PATH, f))]
for datafile in Datasets:
    df = pd.read_csv(MY_PATH + datafile)
    elo_list = df['elo'].values.tolist()
    mmr_list = df['CalculatedMMR'].values.tolist()
    for i in range(len(elo_list)):
        elo_string = split_elo(elo_list[i])
        if 'Unknown' in elo_string:
            newMMR = 'Unknown'
        else:
            newMMR = calculate_player_MMR(elo_string)
        previous = df['CalculatedMMR'][i]
        df.loc[[i],['CalculatedMMR']] = newMMR

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv(MY_PATH + datafile,index=False)
    print(MY_PATH + datafile, "is updated")

# for data in Datasets:
#    df = pd.read_csv(MY_PATH + data)
#    print(df)
