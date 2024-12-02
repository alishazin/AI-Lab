import csv
import pandas as pd

def train(X, y, positive):

    h = pd.Series(dtype='object')

    for index in range(X.shape[0]):

        if y[index] == positive:

            if h.empty:
                h = pd.Series(X.loc[index], index=X.columns)
            
            else:

                for col_index in range(X.shape[1]):

                    if X.loc[index][col_index] != h[col_index]:
                        h[col_index] = '?'

    return h
            

df = pd.read_csv("1.csv")

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

h = train(X, y, "Yes")

print(h)