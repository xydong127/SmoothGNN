DATADIR = '../datasets'
  
TFINANCE = 'tfinance'
TSOCIAL = 'tsocial'
YELP = 'yelp'
AMAZON = 'amazon'
DGRAPHFIN = 'dgraphfin'
ELLIPTIC = 'elliptic'
REDDIT = 'reddit'
TOLOKERS = 'tolokers'
QUESTIONS = 'questions'

INDEX = '_index.txt'

SMALL = 'small'
MEDIUM = 'medium'
LARGE = 'large'

DATASETS = {REDDIT: SMALL, 
            TOLOKERS: SMALL,
            AMAZON: SMALL,
            TFINANCE: MEDIUM,
            YELP: MEDIUM,
            QUESTIONS: MEDIUM,
            ELLIPTIC: LARGE,
            DGRAPHFIN: LARGE,
            TSOCIAL: LARGE}

PARAMETERS = {SMALL: [0.0001, 4, 0.05, 97, 0],
              MEDIUM: [0.0005, 5, 0.01, 43, 4e-3],
              LARGE: [0.0005, 6, 0.05, 23, 4e-3]}

def set_paras(data):
    return PARAMETERS[DATASETS[data]]

