import pandas as pd
from riotwatcher import LolWatcher, ApiError

import EloDictionary as mmrDic

#"RGAPI-12ef572c-2b1d-4c66-875b-5b4fe66fd147"
#"RGAPI-03942335-edb3-49cf-8d03-926fd5e213ae"
#"RGAPI-37a37004-6ace-46c5-8704-d04477882120"
#"RGAPI-026b84df-d6f1-4822-b5e2-0d65e0a3704b"
RIOT_API = "RGAPI-329484ba-8b3d-40d6-8efb-260f90ff7977"
#"RGAPI-54f8c842-80dc-4d62-b07f-c78cb93ed7f8"
WATCHER = LolWatcher(RIOT_API)
MY_REGION = "EUW1"

###ESTIMACION DE PUNTOS
###TIER 100,    200,    300,   400, 500,     600    ,  700,    800,         900   [0-200] [201-600]   [601- ...]
###TIER HIERRO, BRONCE, PLATA, ORO, PLATINO, DIAMANTE, MASTER, GRANDMASTER, CHALLENGER

###RANK IV, III, II, I
###RANK 20, 40, 60, 80
#319
#300+

#Diamante 1 100 lps -> 600 + 80 + 10 = 690
#Master 0 lps -> 700
#Grandmaster 500lps -> 1300
#Challenger 600 lps
###LeaguePoints /10 y habria que sumarlos [0,100]

###Conversion MMR Tier * Rank + LP


def get_player_by_account_id(account_id):
    player_stats = WATCHER.league.by_id(MY_REGION, account_id)
    return get_player_rankedStats(player_stats)


def get_player_rankedInfo(summoner_id):
    player_stats = WATCHER.league.by_summoner(MY_REGION, summoner_id)
    if not player_stats:
        return "Unranked"
    else:
        return get_player_rankedStats(player_stats)


def get_player_SummonerInfo(summoner_name):
    try:
        summoner_info = WATCHER.summoner.by_name(MY_REGION, summoner_name)
        return summoner_info
    except ApiError as err:
        if err.response.status_code == 429:
            print('We should retry in {} seconds.'.format(err.headers['Retry-After']))
            print('this retry-after is handled by default by the RiotWatcher library')
            print('future requests wait until the retry-after time passes')
        elif err.response.status_code == 404:
            print('A problem with a summoner\'s  ridiculous was found...')
            print('Ignoring this match...')

        else:
            raise
        return "Unknown data"


def get_player_rankedStats(player_stats):
    for rows in player_stats:
        if rows['queueType'] == 'RANKED_SOLO_5x5':
            return rows
    return "Unranked"


def calculate_player_MMR(elo):
    mmr_score = 0
    mmr_score += mmrDic.switch_tier.get(elo[0], elo[0])()
    mmr_score += mmrDic.switch_rank.get(elo[1], elo[1])()
    mmr_score += int(elo[2])/10
    return mmr_score


def get_match_info(match_id, path):
    match_stats = WATCHER.match.by_id(MY_REGION, match_id)
    players = []
    success_game = True
    for p in match_stats['participantIdentities']:
        player_row = {}
        #################SUMMONER BASIC INFO ########################
        summoner_name = player_row['summonerName'] = p['player']['summonerName']
        summoner_info = get_player_SummonerInfo(summoner_name)
        if summoner_info == "Unknown data":
            success_game = False
            break
        else:
            player_row['summonerLevel'] = summoner_info['summonerLevel']

            #################SUMMONER RANKED INFO ########################
            ranked_info = get_player_rankedInfo(summoner_info['id'])
            if ranked_info == 'Unranked':
                player_row['elo'] = "Unknown"
                player_row['hot_streak'] = "Unknown"
                player_row['CalculatedMMR'] = "Unknown"
            else:
                player_row['elo'] = (ranked_info['tier'], ranked_info['rank'], ranked_info['leaguePoints'])
                player_row['hot_streak'] = ranked_info['hotStreak']
                player_row['CalculatedMMR'] = calculate_player_MMR(player_row['elo'])

            #################MATCH INFO ########################ç
            player_row['match_id'] = match_id
            player_row['GameDuration'] = match_stats['gameDuration']
            player_row['Victory'] = match_stats['participants'][int(p['participantId']) - 1]['stats']['win']

            players.append(player_row)

    if success_game:
        df = pd.DataFrame(players)
        df.to_csv(path,index=False)
        print(df)

    return success_game
