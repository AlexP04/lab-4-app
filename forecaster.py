import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf
from sklearn.ensemble import IsolationForest
RANDOM_STATE = 42

danger_levels = [
    (11.7, 10.5),
    (1, 0),
    (11.7, 10.5)
]

TIME_DELTA = 10

__author__ = 'forteraiger'

class Forecaster:
    def __init__(self, solver):
        self.solver = solver
        lamb = list()
        for i in range(self.solver.Y.shape[1]):  # `i` is an index for Y
            lamb_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                lamb_i_j = list()
                for k in range(self.solver.dim[j]):  # `k` is an index for vector component
                    lamb_i_j_k = self.solver.Lamb[shift:shift + self.solver.deg[j], i].ravel()
                    shift += self.solver.deg[j]
                    lamb_i_j.append(lamb_i_j_k)
                lamb_i.append(lamb_i_j)
            lamb.append(lamb_i)

        self.lamb = lamb
        self.a = solver.a.T.tolist()
        self.c = solver.c.T.tolist()
        self.X_min = self.solver.datas[:, :-self.solver.Y.shape[1]].min(axis=0)
        self.X_max = self.solver.datas[:, :-self.solver.Y.shape[1]].max(axis=0)
        self.Y_min, self.Y_max = solver.Y_.min(axis=0), solver.Y_.max(axis=0)

    def forecast(self, X, form='additive'):
        Y = np.zeros((len(X), self.solver.Y.shape[1]))
        n, m = X.shape
        vec = np.zeros_like(X)
        for j in range(m):
            minv = np.min(X[:,j])
            maxv = np.max(X[:,j])
            for i in range(n):
                if np.allclose(maxv - minv, 0):
                    vec[i,j] = X[i,j] is X[i,j] != 0
                else:
                    vec[i,j] = (X[i,j] - minv)/(maxv - minv)
        X_norm = np.array(vec)

        def evalF(x, i):
            if form == 'additive':
                res = 0
            elif form == 'multiplicative':
                res = 1   
            for j in range(3):
                for k in range(len(self.lamb[i][j])):
                    shift = sum(self.solver.dim[:j]) + k
                    for n in range(len(self.lamb[i][j][k])):
                        coef = self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n]
                        if form == 'additive':
                            res += coef * self.solver.poly_f(n, x[:, shift])
                        elif form == 'multiplicative':
                            res += (1 + self.solver.poly_f(n, x[:, shift]) + 1e-8) ** coef
            if form == 'additive':
                return res
            elif form == 'multiplicative':
                return res - 1  

        res = np.array([evalF(X_norm, i) for i in range(Y.shape[1])]).T
        return res * (self.Y_max - self.Y_min) + self.Y_min
    

def ARIMAX(endog, order=None, exog=None):
    if order is None:
        order = (0, 0, 0)
    if len(order) == 3:
        p, q, d = order
    elif len(order) == 2:
        p, q = order
        d = 0
    if p == 0:
        pacf_tolerance = 1.96 / np.sqrt(len(endog))
        try:
            p = np.where(abs(pacf(endog, nlags=10)) >= pacf_tolerance)[0].max()+1
        except:
            p = 0
    if q == 0:
        ma = ewma(endog)
        pacf_tolerance = 1.96 / np.sqrt(len(endog))
        try:
            q = np.where(abs(pacf(ma, nlags=10)) >= pacf_tolerance)[0].max()+1
        except:
            q = 0

    model = ARIMA(endog, exog, (p, q, d))
    model = model.fit()
    return model

def FaultProb(y, y_emergency, y_fatal, window_size):
    y_ma = pd.DataFrame(y).rolling(window_size).mean().values.flatten()
    res = (y_emergency - y_ma) / (y_emergency - y_fatal)
    res[res > 1] = 1
    res[res < 0] = 0
    return res

def ClassifyEmergency(y1, y2, y3):
    res = []
    # 10.5 < y1 <= 11.7
    if danger_levels[0][1] < y1 <= danger_levels[0][0]:
        res.append('низька напруга в бортовій мережі')
    elif y1 <= danger_levels[0][1]:
        res.append('критично низька напруга в бортовій мережі')
    if danger_levels[1][1] < y2 <= danger_levels[1][0]:
        res.append('мала кількість пального')
    elif y2 <= danger_levels[1][1]:
        res.append('немає пального')
    if danger_levels[2][1] < y3 <= danger_levels[2][0]:
        res.append('низька напруга в АКБ')
    elif y3 <= danger_levels[2][1]:
        res.append('критично низька напруга в АКБ')
    
    if len(res) > 0:
        return ', '.join(res)
    else:
        return '-' + ' '*40

def ClassifyState(y1, y2, y3):
    if (
        danger_levels[0][1] < y1 <= danger_levels[0][0] or
        danger_levels[1][1] < y2 <= danger_levels[1][0] or
        danger_levels[2][1] < y3 <= danger_levels[2][0]):
        return 'Нештатна ситуація'
    elif (
        y1 <= danger_levels[0][1] or 
        y2 <= danger_levels[1][1] or 
        y3 <= danger_levels[2][1]):
        return 'Аварійна ситуація'
    else:
        return 'Нормальний стан'

def AcceptableRisk(y_slice, danger_levels):
    deltas = np.diff(y_slice, axis=0).max(axis=0)
    y_to_danger = np.array([
        y_slice[-1][i] - danger_levels[i][1]
        for i in range(len(danger_levels))
    ])
    return max((y_to_danger / deltas).min(), 0)

def CheckSensors(x):
    all_pred = []
    for i in range(x.shape[1]):
        clf = IsolationForest(
            random_state=RANDOM_STATE
        ).fit(x[:, i].reshape(-1, 1))
        pred = clf.predict(x[:, i].reshape(-1, 1))
        all_pred.append(pred)
    all_pred = np.array(all_pred)
    return 1 * (all_pred.sum(axis=0) == -2)

def highlight(s, column, vals, colors):
    for val, color in zip(vals, colors):
        if s[column] == val:
            return [f'background-color: {color}'] * len(s)
    else:
        return [f'background-color: white'] * len(s)

def ewma(x, alpha=None):
    if alpha is None:
        alpha = 2 / (len(x) + 1)
    x = np.array(x)
    n = x.size

    w0 = np.ones(shape=(n, n)) * (1 - alpha)
    p = np.vstack([np.arange(i, i-n, -1) for i in range(n)])

    w = np.tril(w0**p,0)

    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)