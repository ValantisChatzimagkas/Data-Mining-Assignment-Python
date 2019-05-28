import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


transpose = input('Transpose DataSet ? (Y/N): ')

if transpose == 'Y':
    pd.read_csv('DataSet.csv', header=None).T.to_csv('T_DataSet.csv', header=False, index=False)
    print('Transposed dataset created successfully!!')
    print('\n')
    dataset = pd.read_csv('T_DataSet.csv')
    NE1 = [3, np.nan, 5, 4, 2, 3, np.nan, 5, 4, 2, 3, np.nan, 5, 4, 2, 3, np.nan, 5, 4, 2]
    NE2 = [np.nan, 5, 2, 2, 4, np.nan, 1, 3, 2, 1, np.nan, 5, 2, 2, 4, np.nan, 1, 3, 2, 1]

if transpose == 'N':
    dataset = pd.read_csv('DataSet.csv')
    NE1 = [3, np.nan, 5, 4, 2, 3, np.nan, 5]
    NE2 = [np.nan, 5, 2, 2, 4, np.nan, 1, 3]


col1 = dataset.iloc[:,:1].values

data = dataset.iloc[:,1:].values

HEADERS = list(dataset.head(0))
del HEADERS[0]

k = int(input('Give k for k-nn :'))
print('\n')

print('Give Ratings for new entry...')
NE0 = list()
for h in range(len(HEADERS)):
    input_value = input('Rating for '+HEADERS[h]+' : ')
    if input_value != '':
        input_value = int(input_value)
    else:
        input_value = np.nan
    NE0.append(input_value)

print('\n')
print('NE0 Ratings:')
print(NE0)
print('\n')

print('NE1 Ratings:')
print(NE1)
print('\n')

print('NE2 Ratings:')
print(NE2)
print('\n')

"""STRATEGY: 
    none = np.nan at missing values
    mean = mean at missing values
"""
STRATEGY = 'none'

if STRATEGY == 'mean':
    imput = SimpleImputer(missing_values=np.nan, strategy='mean')
    imput = imput.fit(data)
    data = imput.transform(data)


def pearson_correlation(numbers_x, numbers_y):
    bad = ~np.logical_or(np.isnan(numbers_x), np.isnan(numbers_y))
    numbers_x = np.compress(bad, numbers_x)
    numbers_y = np.compress(bad, numbers_y)
    x = numbers_x - numbers_x.mean()
    y = numbers_y - numbers_y.mean()
    return (x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum())


def user_correlation_array(d, u):
    rne = list()
    for i in range(len(d)):
        rne.append(pearson_correlation(d[i], u))
    return rne


rNE0 = user_correlation_array(data, NE0)
rNE1 = user_correlation_array(data, NE1)
rNE2 = user_correlation_array(data, NE2)

print('ALL CORREL VALUES r for NE0')
print(rNE0)
print('\n')

print('ALL CORREL VALUES r for NE1')
print(rNE1)
print('\n')

print('ALL CORREL VALUES r for NE2')
print(rNE2)
print('\n')


def get_positions_from_correlation_list(u, k):
    theset = frozenset(u)
    theset = sorted(theset, reverse=True)
    positions = []
    for j in range(k):
        for i in range(len(u)):
            if u[i] == theset[j]:
                positions.append(i)
    return positions[:k]


print(str(k) + ' K-NN POSITIONS OF CORREL VALUES')
print(get_positions_from_correlation_list(rNE0, k))
print(get_positions_from_correlation_list(rNE1, k))
print(get_positions_from_correlation_list(rNE2, k))
print('\n')


def get_values_from_correlation_list(u, k):
    values = list()
    positions = get_positions_from_correlation_list(u, k)
    for i in range(len(positions)):
        values.append(u[positions[i]])
    return values


print(str(k) + ' K-NN CORREL VALUES')
print(get_values_from_correlation_list(rNE0, k))
print(get_values_from_correlation_list(rNE1, k))
print(get_values_from_correlation_list(rNE2, k))
print('\n')


def get_values_from_data_list(m, u, k):
    values = list()
    positions = get_positions_from_correlation_list(u, k)
    for i in range(len(positions)):
         values.append(m[positions[i]])
    return values


def find_nan_positions(l):
    nans = list()
    for i in range(len(l)):
        if np.isnan(l[i]):
            nans.append(i)
    return nans


print('Calculated rating with user-based collaborative filtering')


def divide(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


def get_calculated_rating(m, u, k):
    nan_positions = find_nan_positions(get_values_from_data_list(m, u, k))
    a = get_values_from_correlation_list(u, k)
    b = get_values_from_data_list(m, u, k)

    if len(nan_positions) > 0:
        for x in reversed(range(len(a))):
                if x in nan_positions:
                    del a[x]
        for x in reversed(range(len(b))):
                if x in nan_positions:
                    del b[x]

    c = np.multiply(a, b)
    return divide(sum(c), sum(a))


# print('NE1, The DaVinci Code : {} ' . format(get_calculated_rating(dataset['THE DA VINCI CODE'], rNE1, k)))
# print('NE1, RUNNY BABBIT : {} ' . format(get_calculated_rating(dataset['RUNNY BABBIT'], rNE1, k)))
# print('\n')
#
# print('NE2, TRUE BELIEVER : {} ' . format(get_calculated_rating(dataset['TRUE BELIEVER'], rNE2, k)))
# print('NE2, THE KITE RUNNER : {} ' . format(get_calculated_rating(dataset['THE KITE RUNNER'], rNE2, k)))
# print('\n')


def calculated_rating_list_by_user(b, u, k):
    l = list()
    for i in range(len(b)):
        l.append(get_calculated_rating(dataset[b[i]], u, k))
    return l


print('calculated rating list by user NE0')
print(calculated_rating_list_by_user(HEADERS, rNE0, k))
print('\n')

print('calculated rating list by user NE1')
print(calculated_rating_list_by_user(HEADERS, rNE1, k))
print('\n')

print('calculated rating list by user NE2')
print(calculated_rating_list_by_user(HEADERS, rNE2, k))
print('\n')


def get_mae_by_user(b, u, uc, k):
    nan_positions = find_nan_positions(u)
    a = calculated_rating_list_by_user(b, uc, k)
    b = u

    if len(nan_positions) > 0:
        for x in reversed(range(len(a))):
                if x in nan_positions:
                    del a[x]
        for x in reversed(range(len(b))):
                if x in nan_positions:
                    del b[x]

    return np.mean(list(abs(np.array(b) - np.array(a))))


print('MAE NE0')
print(get_mae_by_user(HEADERS, NE0, rNE0, k))
print('\n')

print('MAE NE1')
print(get_mae_by_user(HEADERS, NE1, rNE1, k))
print('\n')

print('MAE NE2')
print(get_mae_by_user(HEADERS, NE2, rNE2, k))
print('\n')
