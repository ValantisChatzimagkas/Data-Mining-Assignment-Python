import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

"""DATASET"""
dataset = pd.read_csv('DataSet.csv')

"""USERS"""
users = dataset.iloc[:,:1].values

"""DATA"""
data = dataset.iloc[:,1:].values

"""BOOKS"""
BOOKS = list(dataset.head(0))
del BOOKS[0]

"""NEW USERS"""
NU1 = [3, np.nan, 5, 4, 2, 3, np.nan, 5]
NU2 = [np.nan, 5, 2, 2, 4, np.nan, 1, 3]

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
    rnu = list()
    for i in range(len(d)):
        rnu.append(pearson_correlation(d[i], u))
    return rnu


rNU1 = user_correlation_array(data, NU1)
rNU2 = user_correlation_array(data, NU2)

print('ALL CORREL VALUES r for NU1')
print(rNU1)
print('\n')

print('ALL CORREL VALUES r for NU2')
print(rNU2)
print('\n')


def get_positions_from_correlation_list(u):
    theset = frozenset(u)
    theset = sorted(theset, reverse=True)
    thedict = {}
    for j in range(3):
        positions = [i for i, x in enumerate(u) if x == theset[j]]
        thedict[theset[j]] = positions
    output = thedict.get(theset[0]) + thedict.get(theset[1]) + thedict.get(theset[2])
    return output[0:3]


print('3 POSITIONS OF CORREL VALUES')
print(get_positions_from_correlation_list(rNU1))
print(get_positions_from_correlation_list(rNU2))
print('\n')


def get_values_from_correlation_list(u):
    values = list()
    positions = get_positions_from_correlation_list(u)
    for i in range(len(positions)):
        values.append(u[positions[i]])
    return values


print('3 CORREL VALUES')
print(get_values_from_correlation_list(rNU1))
print(get_values_from_correlation_list(rNU2))
print('\n')


def get_values_from_movie_list(m, u):
    values = list()
    positions = get_positions_from_correlation_list(u)
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


def get_calculated_rating(m, u):
    nan_positions = find_nan_positions(get_values_from_movie_list(m, u))
    a = get_values_from_correlation_list(u)
    b = get_values_from_movie_list(m, u)

    if len(nan_positions) > 0:
        for x in reversed(range(len(a))):
                if x in nan_positions:
                    del a[x]
        for x in reversed(range(len(b))):
                if x in nan_positions:
                    del b[x]

    c = np.multiply(a, b)
    return divide(sum(c), sum(a))


print('NU1, The DaVinci Code : {} ' . format(get_calculated_rating(dataset['THE DA VINCI CODE'], rNU1)))
print('NU1, RUNNY BABBIT : {} ' . format(get_calculated_rating(dataset['RUNNY BABBIT'], rNU1)))
print('\n')

print('NU2, TRUE BELIEVER : {} ' . format(get_calculated_rating(dataset['TRUE BELIEVER'], rNU2)))
print('NU2, THE KITE RUNNER : {} ' . format(get_calculated_rating(dataset['THE KITE RUNNER'], rNU2)))
print('\n')


def calculated_rating_list_by_user(b, u):
    l = list()
    for book in range(len(b)):
        l.append(get_calculated_rating(dataset[b[book]], u))
    return l


print('calculated rating list by user NU1')
print(calculated_rating_list_by_user(BOOKS, rNU1))
print('\n')

print('calculated rating list by user NU2')
print(calculated_rating_list_by_user(BOOKS, rNU2))
print('\n')


def get_mae_by_user(b, u, uc):
    nan_positions = find_nan_positions(u)
    a = calculated_rating_list_by_user(b, uc)
    b = u

    if len(nan_positions) > 0:
        for x in reversed(range(len(a))):
                if x in nan_positions:
                    del a[x]
        for x in reversed(range(len(b))):
                if x in nan_positions:
                    del b[x]

    return np.mean(list(abs(np.array(b) - np.array(a))))


print('MAE NU1')
print(get_mae_by_user(BOOKS, NU1, rNU1))
print('\n')

print('MAE NU2')
print(get_mae_by_user(BOOKS, NU2, rNU2))
print('\n')
