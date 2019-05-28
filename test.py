import pandas as pd

pd.read_csv('DataSet.csv', header=None).T.to_csv('T_DataSet.csv', header=False, index=False)
