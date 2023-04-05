from copy import deepcopy
import numpy as np
import pandas as pd
from openpyxl import Workbook
from tabulate import tabulate as tb
from datetime import datetime
from tqdm import tqdm
from scipy import special
from scipy.sparse.linalg import cg



func_runtimes = {}

def profile(method):
    """ Profiling decorator. """
    def wrapper(*args, **kw):
        start_time = datetime.now()
        result = method(*args, **kw)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        func_runtimes.setdefault(method.__name__, []).append(elapsed_time)
        return result
    return wrapper  # Decorated method (need to return this).

def sigmoid(x):
    return np.tanh(x/2)/2 + 0.5


class Solve(object):
    OFFSET = 1e-10

    def __init__(self, d):
        self.dim = d['dimensions']
        self.filename_input = d['input_file']
        self.filename_output = d['output_file']
        self.deg = list(map(lambda x: x + 1, d['degrees']))  # on 1 more because include 0
        self.weights = d['weights']
        self.poly_type = d['poly_type']
        self.splitted_lambdas = d['lambda_multiblock']
        self.norm_error = 0.0
        self.eps = 1E-8
        self.error = 0.0

    # @profile
    def define_data(self):
        # all data from file_input in float
        # self.datas = np.fromstring(self.filename_input, sep='\t').reshape(-1, sum(self.dim))
        self.datas = self.filename_input.copy()
        self.n = len(self.datas)
        # list of sum degrees [ 3,1,2] -> [3,4,6]
        self.dim_integral = [sum(self.dim[:i + 1]) for i in range(len(self.dim))]

    # @profile
    def _minimize_equation(self, A, b):
        """
        Finds such vector x that |Ax-b|->min.
        :param A: Matrix A
        :param b: Vector b
        :return: Vector x
        """
        return cg(A.T @ A, A.T @ b, tol=self.eps)[0].reshape(-1,1)

    # @profile
    def norm_data(self):
        """
        norm vectors value to value in [0,1]
        :return: float number in [0,1]
        """
        n,m = self.datas.shape
        vec = np.ndarray(shape=(n,m),dtype=float)
        for j in range(m):
            minv = np.min(self.datas[:,j])
            maxv = np.max(self.datas[:,j])
            for i in range(n):
                if np.allclose(maxv - minv, 0):
                    vec[i,j] = self.datas[i,j] is self.datas[i,j] != 0
                else:
                    vec[i,j] = (self.datas[i,j] - minv)/(maxv - minv)
        self.data = np.array(vec)

    # @profile
    def define_norm_vectors(self):
        """
        buile matrix X and Y
        :return:
        """
        X1 = self.data[:, :self.dim_integral[0]]
        X2 = self.data[:, self.dim_integral[0]:self.dim_integral[1]]
        X3 = self.data[:, self.dim_integral[1]:self.dim_integral[2]]
        # matrix of vectors i.e.X = [[X11,X12],[X21],...]
        self.X = [X1, X2, X3]
        self.minX = np.min(self.datas[:, :self.dim_integral[2]], axis=0)
        self.maxX = np.max(self.datas[:, :self.dim_integral[2]], axis=0)
        self.minY = np.min(self.datas[:, self.dim_integral[2]:], axis=0)
        self.maxY = np.max(self.datas[:, self.dim_integral[2]:], axis=0)
        # number columns in matrix X
        self.mX = self.dim_integral[2]
        # matrix, that consists of i.e. Y1,Y2
        self.Y = self.data[:, self.dim_integral[2]:self.dim_integral[3]]
        self.Y_ = self.datas[:, self.dim_integral[2]:self.dim_integral[3]]
        self.X_ = [self.datas[:, :self.dim_integral[0]], self.datas[:, self.dim_integral[0]:self.dim_integral[1]],
                   self.datas[:, self.dim_integral[1]:self.dim_integral[2]]]

    # @profile
    def built_B(self):
        def B_average():
            """
            Vector B as average of max and min in Y. B[i] = max Y[i,:]
            :return:
            """
            b = np.tile((self.Y.max(axis=1) + self.Y.min(axis=1)) / 2, (self.dim[3], 1)).T
            return b

        def B_scaled():
            """
            Vector B  = Y
            :return:
            """
            return deepcopy(self.Y)

        if self.weights == 'Середнє арифметичне':
            self.B = B_average()
        elif self.weights == 'Нормоване значення':
            self.B = B_scaled()
        else:
            exit('B not defined')
        self.B_log = np.log(self.B + 1 + self.OFFSET)

    # @profile
    def poly_func(self):
        """
        Define function to polynomials
        :return: function
        """
        if self.poly_type =='Chebyshev':
            self.poly_f = special.eval_sh_chebyt
        elif self.poly_type == 'Legandre':
            self.poly_f = special.eval_sh_legendre
        elif self.poly_type == 'Lagger':
            self.poly_f = special.eval_laguerre
        elif self.poly_type == 'Hermitt':
            self.poly_f = special.eval_hermite

    # @profile
    def built_A(self):
        """
        built matrix A on shifted polynomials Chebysheva
        :param self.p:mas of deg for vector X1,X2,X3 i.e.
        :param self.X: it is matrix that has vectors X1 - X3 for example
        :return: matrix A as ndarray
        """

        def coordinate(v, deg):
            """
            :param v: vector
            :param deg: chebyshev degree polynom
            :return:column with chebyshev value of coordiate vector
            """
            c = np.ndarray(shape=(self.n, 1), dtype=float)
            for i in range(self.n):
                c[i, 0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            """
            :param vec: it is X that consist of X11, X12, ... vectors
            :param p: max degree for chebyshev polynom
            :return: part of matrix A for vector X1
            """
            n, m = vec.shape
            a = np.ndarray(shape=(n, 0), dtype=float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:, j], i)
                    a = np.append(a, ch, 1)
            return a

        A = np.ndarray(shape=(self.n, 0), dtype=float)
        for i in range(len(self.X)):
            vec = vector(self.X[i], self.deg[i])
            A = np.append(A, vec, 1)
        # self.A = np.matrix(A)
        self.A = np.array(A)
        self.A_log = np.log((self.A + 1 + self.OFFSET))

    # @profile
    def lamb(self):
        lamb = np.ndarray(shape=(self.A.shape[1], 0), dtype=float)
        for i in range(self.dim[3]):
            if self.splitted_lambdas:
                boundary_1 = self.deg[0] * self.dim[0]
                boundary_2 = self.deg[1] * self.dim[1] + boundary_1
                lamb1 = self._minimize_equation(self.A_log[:, :boundary_1], self.B_log[:, i])
                lamb2 = self._minimize_equation(self.A_log[:, boundary_1:boundary_2], self.B_log[:, i])
                lamb3 = self._minimize_equation(self.A_log[:, boundary_2:], self.B_log[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3)), axis=1)
            else:
                lamb = np.append(lamb, self._minimize_equation(self.A_log, self.B_log[:, i]), axis=1)
        self.Lamb = np.array(lamb)

    # @profile
    def psi(self):
        def built_psi(lamb):
            """
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            """
            psi = np.ndarray(shape=(self.n, self.mX), dtype=float)
            q = 0  # iterator in lamb and A
            l = 0  # iterator in columns psi
            for k in range(len(self.X)):  # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):  # choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                        psi[i, l] = self.A_log[i, q:q + self.deg[k]] @ lamb[q:q + self.deg[k]]
                    q += self.deg[k]
                    l += 1
            return np.array(psi)

        self.Psi_log = []  # as list because psi[i] is matrix(not vector)
        self.Psi = []
        for i in range(self.dim[3]):
            self.Psi_log.append(built_psi(self.Lamb[:, i]))
            self.Psi.append(np.exp(self.Psi_log[i]) - 1 - self.OFFSET)

    # @profile
    def built_a(self):
        self.a = np.ndarray(shape=(self.mX, 0), dtype=float)
        for i in range(self.dim[3]):
            a1 = self._minimize_equation(self.Psi_log[i][:, :self.dim_integral[0]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a2 = self._minimize_equation(self.Psi_log[i][:, self.dim_integral[0]:self.dim_integral[1]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a3 = self._minimize_equation(self.Psi_log[i][:, self.dim_integral[1]:],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            self.a = np.append(self.a, np.vstack((a1, a2, a3)), axis=1)

    # @profile
    def built_F1i(self, psi, a):
        """
        not use; it used in next function
        :param psi: matrix psi (only one
        :param a: vector with shape = (6,1)
        :param dim_integral:  = [3,4,6]//fibonacci of deg
        :return: matrix of (three) components with F1 F2 and F3
        """
        m = len(self.X)  # m  = 3
        F1i = np.ndarray(shape=(self.n, m), dtype=float)
        k = 0  # point of beginning column to multiply
        for j in range(m):  # 0 - 2
            for i in range(self.n):  # 0 - 49
                F1i[i, j] = psi[i, k:self.dim_integral[j]] @ a[k:self.dim_integral[j]]
            k = self.dim_integral[j]
        return np.array(F1i)

    # @profile
    def built_Fi(self):
        self.Fi_log = []
        self.Fi = []
        for i in range(self.dim[3]):
            self.Fi_log.append(self.built_F1i(self.Psi_log[i], self.a[:, i]))
            self.Fi.append(np.exp(self.Fi_log[-1]) - 1 - self.OFFSET)

    # @profile
    def built_c(self):
        self.c = np.ndarray(shape=(len(self.X), 0), dtype=float)
        for i in range(self.dim[3]):
            self.c = np.append(self.c, self._minimize_equation(self.Fi_log[i], np.log(self.Y[:, i] + 1 + self.OFFSET)), axis=1)

    # @profile
    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype=float)
        for j in range(F.shape[1]):  # 2
            for i in range(F.shape[0]):  # 50
                F[i, j] = self.Fi_log[j][i, :] @ self.c[:, j]
        self.F_log = np.array(F)
        self.F = np.exp(self.F_log) - 1
        self.norm_error = np.abs(self.Y - self.F).max(axis=0).tolist()

    # @profile
    def built_F_(self):
        minY = self.Y_.min(axis=0)
        maxY = self.Y_.max(axis=0)
        self.F_ = np.multiply(self.F, maxY - minY) + minY

        self.error = np.abs(self.Y_ - self.F_).max(axis=0).tolist()

     # @profile
    def save_to_file(self):
        # wb = Workbook()
        #get active worksheet
        # ws = wb.active
        # ws = []
        ws_dict = {}

        l = [None]

        # ws.append(['Вхідні дані: X'])
        ws = []
        for i in range(self.n):
            ws.append(l+self.datas[i,:self.dim_integral[3]].tolist())
        ws_dict['Вхідні дані: X'] = ws

        # ws.append(['Вхідні дані: Y'])
        ws = []
        for i in range(self.n):
            ws.append(l+self.datas[i,self.dim_integral[2]:self.dim_integral[3]].tolist())
        ws_dict['Вхідні дані: Y'] = ws

        # ws.append(['X нормалізовані:'])
        ws = []
        for i in range(self.n):
            ws.append(l+self.data[i,:self.dim_integral[2]].tolist())
        ws_dict['X нормалізовані:'] = ws

        # ws.append(['Y нормалізовані:'])
        ws = []
        for i in range(self.n):
            ws.append(l+self.data[i,self.dim_integral[2]:self.dim_integral[3]].tolist())
        ws_dict['Y нормалізовані:'] = ws

        # ws.append(['Матриця Lambda:'])
        ws = []
        for i in range(self.Lamb.shape[0]):
            ws.append(l+self.Lamb[i].tolist())
        ws_dict['Матриця Lambda:'] = ws

        for j in range(len(self.Psi)):
            s = 'Матриця Psi%i:' %(j+1)
            #  ws.append([s])
            ws = []
            for i in range(self.n):
                ws.append(l+self.Psi[j][i].tolist())
            ws_dict[s] = ws

        # ws.append(['Матриця a:'])
        ws = []
        for i in range(self.mX):
             ws.append(l+self.a[i].tolist())
        ws_dict['Матриця a:'] = ws

        for j in range(len(self.Fi)):
            s = 'Матриця F%i:' %(j+1)
            #  ws.append([s])
            ws = []
            for i in range(self.Fi[j].shape[0]):
                ws.append(l+self.Fi[j][i].tolist())
            ws_dict[s] = ws

        # ws.append(['Матриця c:'])
        ws = []
        for i in range(len(self.X)):
             ws.append(l+self.c[i].tolist())
        ws_dict['Матриця c:'] = ws

        # ws.append(['Нормалізована похибка (Y - F)'])
        # ws.append(l + self.norm_error)
        ws_dict['Нормалізована похибка (Y - F)'] = self.norm_error

        # ws.append(['Похибка (Y_ - F_))'])
        # ws.append(l+self.error)
        ws_dict['Похибка (Y_ - F_))'] = self.error

        return ws_dict
        # wb.save(self.filename_output)


    # @profile
    def show_streamlit(self):
        res = []
        res.append(('Вхідні дані',
            pd.DataFrame(self.datas, 
            columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.dim[i])] + [f'Y{i+1}' for i in range(self.dim[-1])],
            index = np.arange(1, self.n+1))
        ))
        res.append(('Нормовані вхідні дані',
            pd.DataFrame(self.data, 
            columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.dim[i])] + [f'Y{i+1}' for i in range(self.dim[-1])],
            index = np.arange(1, self.n+1))
        ))

        res.append((r'Матриця $\|\lambda\|$',
            pd.DataFrame(self.Lamb)
        ))
        res.append((r'Матриця $\|a\|$',
            pd.DataFrame(self.a)
        ))
        res.append((r'Матриця $\|c\|$',
            pd.DataFrame(self.c)
        ))

        for j in range(len(self.Psi)):
            res.append((r'Матриця $\|\Psi_{}\|$'.format(j+1),
            pd.DataFrame(self.Psi[j])
        ))
        for j in range(len(self.Fi)):
            res.append((r'Матриця $\|\Phi_{}\|$'.format(j+1),
            pd.DataFrame(self.Fi[j])
        ))
    
        df = pd.DataFrame(self.norm_error).T
        df.columns = np.arange(1, len(self.norm_error)+1)
        res.append((r'Нормалізована похибка',
            df
        ))
        df = pd.DataFrame(self.error).T
        df.columns = np.arange(1, len(self.error)+1)
        res.append((r'Похибка',
            df
        ))
        return res

    def prepare(self):
        self.define_data()
        self.norm_data()
        self.define_norm_vectors()
        self.built_B()
        self.poly_func()
        self.built_A()
        self.lamb()
        self.psi()
        self.built_a()
        self.built_Fi()
        self.built_c()
        self.built_F()
        self.built_F_()
        return func_runtimes


# custom function structure
class SolveCustom(Solve):
    def __init__(self, d, nonlinear_func):
        """
        d - dictionary for Solve init,
        nonlinear_func - iterable with:
            nonlinear_func[0] - callable
            nonlinear_func[1] - string, name of callable [0]
            nonlinear_func[2] - string, name of inverse of callable [0]

            e.g. (np.tanh, 'tanh')
        """
        super().__init__(d)
        self.nonlinear_func = nonlinear_func[0]
        self.nonlinear_func_name =  nonlinear_func[1]
        self.nonlinear_func_inv_name =  nonlinear_func[2]

    def built_A(self):
        """
        built matrix A on shifted polynomials Chebysheva
        :param self.p:mas of deg for vector X1,X2,X3 i.e.
        :param self.X: it is matrix that has vectors X1 - X3 for example
        :return: matrix A as ndarray
        """

        def coordinate(v, deg):
            """
            :param v: vector
            :param deg: chebyshev degree polynom
            :return:column with chebyshev value of coordiate vector
            """
            c = np.ndarray(shape=(self.n, 1), dtype=float)
            for i in range(self.n):
                c[i, 0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            """
            :param vec: it is X that consist of X11, X12, ... vectors
            :param p: max degree for chebyshev polynom
            :return: part of matrix A for vector X1
            """
            n, m = vec.shape
            a = np.ndarray(shape=(n, 0), dtype=float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:, j], i)
                    a = np.append(a, ch, 1)
            return a

        A = np.ndarray(shape=(self.n, 0), dtype=float)
        for i in range(len(self.X)):
            vec = vector(self.X[i], self.deg[i])
            A = np.append(A, vec, 1)
        # the difference with Solve starts here
        self.A_log = np.log(self.nonlinear_func(A) + 1 + self.OFFSET)
        self.A = np.exp(self.A_log)

    def psi(self):
        def built_psi(lamb):
            """
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            """
            psi = np.ndarray(shape=(self.n, self.mX), dtype=float)
            q = 0  # iterator in lamb and A
            l = 0  # iterator in columns psi
            for k in range(len(self.X)):  # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):  # choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                        psi[i, l] = self.A_log[i, q:q + self.deg[k]] @ lamb[q:q + self.deg[k]]
                    q += self.deg[k]
                    l += 1
            return np.array(psi)

        self.Psi = []
        self.Psi_Custom = []
        for i in range(self.dim[3]):
            self.Psi_Custom.append(np.log(
                self.nonlinear_func(built_psi(self.Lamb[:, i])) + 1 + self.OFFSET
            ))
            self.Psi.append(np.exp(self.Psi_Custom[-1]))
            # self.Psi.append(np.exp(built_psi(self.Lamb[:, i])) - 1)  # Psi = exp(sum(lambda*tanh(phi))) - 1
            # self.Psi_Custom.append(self.nonlinear_func(self.Psi[-1]))


    def built_a(self):
        self.a = np.ndarray(shape=(self.mX, 0), dtype=float)
        for i in range(self.dim[3]):
            a1 = self._minimize_equation(self.Psi_Custom[i][:, :self.dim_integral[0]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a2 = self._minimize_equation(self.Psi_Custom[i][:, self.dim_integral[0]:self.dim_integral[1]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a3 = self._minimize_equation(self.Psi_Custom[i][:, self.dim_integral[1]:],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            self.a = np.append(self.a, np.vstack((a1, a2, a3)), axis=1)

    def built_Fi(self):
        self.Fi_Custom = list()
        self.Fi = list()
        for i in range(self.dim[3]):
            self.Fi_Custom.append(np.log(
                self.nonlinear_func(self.built_F1i(self.Psi_Custom[i], self.a[:, i])) + 1 + self.OFFSET
            ))
            self.Fi.append(np.exp(self.Fi_Custom[-1]))
            # self.Fi.append(np.exp(self.built_F1i(self.Psi_Custom[i], self.a[:, i])) - 1)  # Fi = exp(sum(a*tanh(Psi))) - 1
            # self.Fi_Custom.append(self.nonlinear_func(self.Fi[i]))

    def built_c(self):
        self.c = np.ndarray(shape=(len(self.X), 0), dtype=float)
        for i in range(self.dim[3]):
            self.c = np.append(self.c, self._minimize_equation(self.Fi_Custom[i], np.log(self.Y[:, i] + 1 + self.OFFSET)), axis=1)

    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype=float)
        for j in range(F.shape[1]):  # 2
            for i in range(F.shape[0]):  # 50
                F[i, j] = self.Fi_Custom[j][i, :] @ self.c[:, j]
        self.F = np.exp(np.array(F)) - 1 - self.OFFSET  # F = exp(sum(c*tanh(Fi))) - 1
        self.norm_error = np.abs(self.Y - self.F).max(axis=0).tolist()