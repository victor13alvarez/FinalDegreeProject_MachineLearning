import CreateDataset as dataImport
import RiotAPI as League

DatasetPath = "Data/match_"
DataFile = ".csv"


def getMatchesIntoCSV():
    matches_id = dataImport.read_file()
    match_count = 6209

    for match_id in matches_id:
        success_game = League.get_match_info(match_id, DatasetPath + str(match_count) + DataFile)
        if success_game:
            print("\nMatch", match_count, "saved \n")
            match_count += 1
        print("Waiting for time request ... \n")


getMatchesIntoCSV()