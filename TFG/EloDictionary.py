def IRON():
    return 100


def BRONZE():
    return 200


def SILVER():
    return 300


def GOLD():
    return 400


def PLATINUM():
    return 500


def DIAMOND():
    return 600


def MASTER():
    return 700


def GRANDMASTER():
    return 800


def CHALLENGER():
    return 900


def IV():
    return 20


def III():
    return 40


def II():
    return 60


def I():
    return 80


switch_tier = {
    "IRON": IRON,
    "BRONZE": BRONZE,
    "SILVER": SILVER,
    "GOLD": GOLD,
    "PLATINUM": PLATINUM,
    "DIAMOND": DIAMOND,
    "MASTER": MASTER,
    "GRANDMASTER": GRANDMASTER,
    "CHALLENGER": CHALLENGER
}

switch_rank = {
    "IV": IV,
    "III": III,
    "II": II,
    "I": I,
}
