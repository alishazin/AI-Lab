import pandas as pd

def train(X, y, yes):

    h = pd.Series(dtype='object')
    
    for r_index in range(X.shape[0]):

        if y[r_index] == yes:

            if h.empty:
                h = pd.Series(X.loc[r_index], index=X.columns)

            else:

                for c_index in range(X.shape[1]):
                    if X.loc[r_index][c_index] != h[c_index]:
                        h[c_index] = '?'

    return h



df = pd.read_csv('data2.csv')

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

h = train(X, y, "Yes")

print(h)