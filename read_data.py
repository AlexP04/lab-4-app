import pandas as pd
import numpy as np


def read_data(filename):
    data = pd.read_csv(filename, sep =';')
    t = data['Unnamed: 0'].tolist()
    X = np.array(data.iloc[:, 1:])
    f = lambda x: float(x.replace(',','.'))
    t = np.array(list(map(f, t)))
    fT = np.vectorize(f)
    X = fT(X)
    return t, X
