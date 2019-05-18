import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

dataset = pd.read_csv('DataSet.csv')

users = dataset.iloc[:,:1].values   #Column USERS
data = dataset.iloc[:,1:].values    #THE REST DATA

"""NEW USERS"""
NU1 = np.array([3,np.nan,5,4,2,3,np.nan,5])
NU2 = np.array([np.nan,5,2,2,4,np.nan,1,3])

""" Filling with mean value
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(data)
dataset = imputer.transform(data)
print(dataset)
"""


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

print('ALL CORREL VALUES r for NU2')
print(rNU2)

def get_positions_from_correlation_list(u):
    theset = frozenset(u)
    theset = sorted(theset, reverse=True)
    thedict = {}
    for j in range(3):
        positions = [i for i, x in enumerate(rNU1) if x == theset[j]]
        thedict[theset[j]] = positions
    output = thedict.get(theset[0]) + thedict.get(theset[1]) + thedict.get(theset[2])
    return output[0:3]

print('3 POSITIONS OF CORREL VALUES')
print(get_positions_from_correlation_list(rNU1))


def get_values_from_correlation_list(u):
    values = list()
    positions = get_positions_from_correlation_list(u)
    for i in range(len(positions)):
        values.append(u[positions[i]])
    return values

print('3 CORREL VALUES')
print(get_values_from_correlation_list(rNU1))


def get_values_from_movie_list(m, u):
    values = list()
    positions = get_positions_from_correlation_list(u)
    for i in range(len(positions)):
         values.append(m[positions[i]])
    return values


print('THE DA VINCI CODE')
print(get_values_from_movie_list(dataset['THE DA VINCI CODE'], rNU1))

print('RUNNY BABBIT')
print(get_values_from_movie_list(dataset['RUNNY BABBIT'], rNU1))


def find_nan_positions(l):
    nans = list()
    for i in range(len(l)):
        if np.isnan(l[i]):
            nans.append(i)
    return nans

print("THE DA VINCI CODE NaN Values")
print(find_nan_positions(get_values_from_movie_list(dataset['THE DA VINCI CODE'], rNU1)))


print("RUNNY BABBIT NaN Values")
print(find_nan_positions(get_values_from_movie_list(dataset['RUNNY BABBIT'], rNU1)))


print('Calculated rating with user-based collaborative filtering')


def get_calculated_rating(m, u):
    nan_positions = find_nan_positions(get_values_from_movie_list(m, u))
    a = get_values_from_correlation_list(u)
    b = get_values_from_movie_list(m, u)

    if len(nan_positions) > 0:
        for x in range(len(a)):
            if x in nan_positions:
                del a[x]
        for x in range(len(b)):
            if x in nan_positions:
                del b[x]

    c = np.multiply(a, b)
    return sum(c) / sum(a)


print('NU1, The DaVinci Code : {} ' . format(get_calculated_rating(dataset['THE DA VINCI CODE'], rNU1)))
print('NU1, RUNNY BABBIT : {} ' . format(get_calculated_rating(dataset['RUNNY BABBIT'], rNU1)))

