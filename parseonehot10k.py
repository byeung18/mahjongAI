import sqlite3
from sqlite3 import Error
import bz2
import re
import numpy as np
import pandas as pd


# constants: total 79 features
DEALER0, DEALER1, DEALER2, DEALER3 = 0, 1, 2, 3
EASTROUND, SOUTHROUND = 4, 5
DORA = 6
DRAW0 = 40
DRAW1, DRAW2, DRAW3 = 74, 75, 76
DISCARD = 77
OTHERDISC = 111
PON0, PON1, PON2, PON3 = 145, 146, 147, 148
CHI0, CHI1, CHI2, CHI3 = 149, 150, 151, 152
RIICHI0, RIICHI1, RIICHI2, RIICHI3 = 153, 154, 155, 156
RON0, RON1, RON2, RON3 = 157, 158, 159, 160
RYUURYOKU = 161

NUM_FEATURES = 162


def create_connection(db_file):
    """

    :param db_file:
    :return:
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


def splitlog(compressed):
    """
    parse gamelog into individual moves for entry into dataframe
    :param compressed:
    :return:
    """
    log = bz2.decompress(compressed[0])
    split_log = log.split(b'<INIT ')
    games = []
    for item in split_log[1:]:
        # split into individual moves
        item2 = item.split(b'/><')
        # print(item2)
        # print(len(item2))

        start = item2.pop(0)
        # print(start)

        # dealer
        [oya] = re.findall(b'oya=\"\d\"', start)
        oya = oya.strip(b'oya="')
        oya = oya.strip(b'"')
        # print(int(oya))

        # opening hands
        [hai0] = re.findall(b'hai0=\"[\d,]+\"', start)
        hai0 = hai0.strip(b'hai0=')
        hai0 = hai0.strip(b'"')
        hai0 = hai0.split(b',')
        hai0 = [int(x) // 4 for x in hai0]
        # print(hai0)

        [hai1] = re.findall(b'hai1=\"[\d,]+\"', start)
        hai1 = hai1.strip(b'hai1=')
        hai1 = hai1.strip(b'"')
        hai1 = hai1.split(b',')
        hai1 = [int(x) // 4 for x in hai1]
        # print(hai1)

        [hai2] = re.findall(b'hai2=\"[\d,]+\"', start)
        hai2 = hai2.strip(b'hai2=')
        hai2 = hai2.strip(b'"')
        hai2 = hai2.split(b',')
        hai2 = [int(x) // 4 for x in hai2]
        # print(hai2)

        [hai3] = re.findall(b'hai3=\"[\d,]+\"', start)
        hai3 = hai3.strip(b'hai3=')
        hai3 = hai3.strip(b'"')
        hai3 = hai3.split(b',')
        hai3 = [int(x) // 4 for x in hai3]
        # print(hai3)

        [round_wind] = re.findall(b'seed=\"\d', start)
        round_wind = round_wind.strip(b'seed=\"')
        round_wind = int(round_wind)

        [dora] = re.findall(b'[\d]+" ten=', start)
        dora = dora.strip(b'" ten=')
        dora = int(dora) // 4
        # print(dora)

        # moves
        game = [int(oya), round_wind, dora, hai0, hai1, hai2, hai3, item2]
        games.append(game)

    # for game in games:
    #     print(game)
    return games


def parse_games(games):
    df_list = []
    for game in games:
        df = _parse_game(game)
        df_list.append(df)

    game_df = pd.concat(df_list)
    # print(game_df.head(200))
    # print(game_df.tail(10))
    return game_df


def _parse_game(game):
    oya = game[0]
    round_wind = game[1]
    dora = game[2]
    hai0, hai1, hai2, hai3 = game[3], game[4], game[5], game[6]
    moves = game[7]

    if b'mjloggm' in moves[-1]:
        moves.pop()

    data = []
    # initialize all zeros

    for idx in range(len(moves) + 16):
        data.append(np.zeros(NUM_FEATURES, dtype=int))

    if oya == 0:
        data[0][DEALER0] = 1
    elif oya == 1:
        data[0][DEALER1] = 1
    elif oya == 2:
        data[0][DEALER2] = 1
    elif oya == 3:
        data[0][DEALER3] = 1

    if round_wind <= 4:
        data[1][EASTROUND] = 1
    else:
        data[1][SOUTHROUND] = 1

    # dora indicator
    data[2][dora + DORA] = 1

    # opening hand 13 draws
    for idx, tile in enumerate(hai0):
        data[idx + 3][tile + 16] = 1

    for idx, move in enumerate(moves):
        _parse_move(move, idx + 16, data)

    # print(data)
    df = pd.DataFrame(data, dtype=np.int)
    # print(df.tail(10))

    return df

    # df = pd.DataFrame(data, columns=['Deal0', 'Deal1', 'Deal2', 'Deal3',
    #                                  'EastRound', 'SouthRound',
    #                                  'Opening', 'Dora',
    #                                  'Draw0', 'Draw1', 'Draw2', 'Draw3',
    #                                  'Discard0', 'Discard1', 'Discard2', 'Discard3',
    #                                  'Pon0', 'Pon1', 'Pon2', 'Pon3',
    #                                  'Kan0', 'Kan1', 'Kan2', 'Kan3',
    #                                  'Chi0', 'Chi1', 'Chi2', 'Chi3',
    #                                  'Riichi0', 'Riichi1', 'Riichi2', 'Riichi3',
    #                                  'Feed0', 'Feed1', 'Feed2', 'Feed3',
    #                                  'Ron0', 'Ron1', 'Ron2', 'Ron3',
    #                                  'Tsumo0', 'Tsumo1', 'Tsumo2', 'Tsumo3',
    #                                  'Ryuuryoku',
    #                                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #                                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    #                                  '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33'])


def _parse_move(move, idx, data):
    move = move.decode("utf-8")
    # print(move)
    code = move[0]

    if len(move) <= 4:
        if code == 'T':
            # data[idx][DRAW0] = 1
            tile = int(move[1:]) // 4 + DRAW0
            data[idx][tile] = 1
        elif code == 'D':
            # data[idx][DISCARD0] = 1
            tile = int(move[1:]) // 4 + DISCARD
            data[idx][tile] = 1
        elif code in ['U', 'V', 'W']:
            if code == 'U':
                data[idx][DRAW1] = 1
            elif code == 'V':
                data[idx][DRAW2] = 1
            else:
                data[idx][DRAW3] = 1
        elif code in ['E', 'F', 'G']:
            tile = int(move[1:]) // 4 + OTHERDISC
            data[idx][tile] = 1
            # if code == 'E':
            #     data[idx][DISCARD1] = 1
            # elif code == 'F':
            #     data[idx][DISCARD2] = 1
            # else:
            #     data[idx][DISCARD3] = 1

    elif code == 'N':
        # someone called tile

        [who] = re.findall(' who=\"\d', move)
        who = who.strip(' who="')
        who = int(who)

        [meld] = re.findall('m="[\d]+\"', move)
        meld = meld.strip('m="')
        meld = meld.strip('"')
        meld = int(meld)

        # chi
        if meld & 0x4:
            data[idx][CHI0 + who] = 1

            # # t0, t1, t2 = (meld >> 3) & 0x3, (meld >> 5) & 0x3, (meld >> 7) & 0x3
            # base_and_called = meld >> 10
            # base = base_and_called // 3
            # # called = base_and_called % 3
            # base = (base // 7) * 9 + base % 7
            #
            # data[idx][base + OFFSET] = 1
            # data[idx][base + OFFSET + 1] = 1
            # data[idx][base + OFFSET + 2] = 1
            # print(data[idx])
            # print("CHI")
        else:
        # elif meld & 0x18:
            data[idx][PON0 + who] = 1
            # # t4 = (meld >> 5) & 0x3
            # # t0, t1, t2 = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))[t4]
            # base_and_called = meld >> 9
            # base = base_and_called // 3
            # # called = base_and_called % 3
            #
            # data[idx][base + OFFSET] = 1
            # # print(data[idx])
            # # print("PON")
        # else:
        #     data[idx][KAN0 + who] = 1
        #     base_and_called = meld >> 8
        #     base = base_and_called // 4
        #
        #     data[idx][base + OFFSET] = 1
        #     # print("KAN")

    elif move[:5] == "AGARI":

        [who] = re.findall(' who=\"\d', move)
        who = who.strip(' who="')
        who = int(who)

        [fromWho] = re.findall('fromWho=\"\d', move)
        fromWho = fromWho.strip('fromWho="')
        fromWho = int(fromWho)

        # if fromWho == who:
        #     data[idx][TSUMO0 + who] = 1
        # else:
        data[idx][RON0 + who] = 1
            # data[idx][FEED0 + fromWho] = 1

    elif move[:5] == "REACH":
        [who] = re.findall(' who=\"[\d]', move)
        who = who.strip(' who="')
        who = int(who)

        data[idx][RIICHI0 + who] = 1

    elif move[:4] == "DORA":
        [dora] = re.findall('hai=\"[\d]+\"', move)
        dora = dora.strip('hai="')
        dora = dora.strip('"')
        dora = int(dora) // 4 + DORA

        # data[idx][DORA] = 1
        data[idx][dora] = 1

    elif move[:9] == "RYUUKYOKU":
        data[idx][RYUURYOKU] = 1


def main():
    database = "2018.db"

    conn = create_connection(database)
    with conn:
        cur = conn.cursor()
        # cur.execute("SELECT COUNT(*) FROM logs WHERE is_tonpusen = 0 and is_hirosima = 0 and is_processed = 1")
        # print(cur.fetchone())
        cur.execute("SELECT log_content FROM logs WHERE is_tonpusen = 0\
         and is_hirosima = 0 and is_processed = 1 and was_error = 0")

        # 173559 games in 2018, 43607 games in 2019
        rows = cur.fetchall()
        # games1 = splitlog(rows[0])
        # games2 = splitlog(rows[11])
        # games3 = splitlog(rows[21])

        # g1 = parse_games(games1)

        # g2 = parse_games(games2)
        # g3 = parse_games(games3)

        file = "onehot2018_60k.csv"

        for row in rows[50000:60000]:
            game = splitlog(row)
            game_df = parse_games(game)
            game_df.to_csv(file, mode='a')
        #
        # print("done 70k")

        # file = "test2018_80k.csv"
        #
        # game = splitlog(rows[80000])
        # game_df = parse_games(game)
        # game_df.to_csv(file)
        #
        # for row in rows[80001:90000]:
        #     game = splitlog(row)
        #     game_df = parse_games(game)
        #     game_df.to_csv(file, mode='a')
        #
        # print("done 80k")
        #
        # file3 = "test2018_90k.csv"
        #
        # game = splitlog(rows[90000])
        # game_df = parse_games(game)
        # game_df.to_csv(file3)
        #
        # for row in rows[90001:100000]:
        #     game = splitlog(row)
        #     game_df = parse_games(game)
        #     game_df.to_csv(file3, mode='a')
        #
        # print("done 90k")
        #
        # file4 = "test2018_100k.csv"
        #
        # game = splitlog(rows[100000])
        # game_df = parse_games(game)
        # game_df.to_csv(file4)
        #
        # for row in rows[100001:110000]:
        #     game = splitlog(row)
        #     game_df = parse_games(game)
        #     game_df.to_csv(file4, mode='a')
        #
        # print("done 100k")

        # count = 0

        # game = splitlog(rows[10000])
        # game_df = parse_games(game)
        # game_df.to_csv(file)
        #
        # for row in rows[10001:20000]:
        #     game = splitlog(row)
        #     game_df = parse_games(game)
        #     game_df.to_csv(file, mode='a')
        #
        #     # count += 1
        #     if count % 50 == 0:
        #         print("{} logs have been processed".format(count))



if __name__ == '__main__':
    main()
    print("done")