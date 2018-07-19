import numpy as np


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def gini2(actual, pred):
    assert (len(actual) == len(pred))
    ind = np.lexsort((actual, pred))
    a = [actual[i] for i in ind]
    a = np.array(a).cumsum()
    a = a / a[-1]
    b = np.append(a[:-1], [0])

    return 0.5 - sum((a+b)/(len(a)*2.0))



predictions = [0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]
actual = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# predictions = [0.9, 0.3, 0.8, 0.75]
# actual = [1, 0, 1, 1]

print(gini(actual, predictions))
print(gini2(actual, predictions))
print(gini2(actual, actual))
# print(gini_normalized(actual, predictions))