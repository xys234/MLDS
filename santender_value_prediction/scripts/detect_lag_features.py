import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

FEATURES = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
            '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
            'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
            '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
            'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
            '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
            '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
            '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']


def get_pred(data, lag=2):
    d1 = data[FEATURES[:-lag]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = data[FEATURES[lag:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[FEATURES[lag - 2]]
    d2 = d2[d2.pred != 0]
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)



train = pd.read_csv('../data/train.csv.zip')
pred = get_pred(train.iloc[3])