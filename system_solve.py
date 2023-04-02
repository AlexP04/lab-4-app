
import numpy as np

def conjugate_gradient_method(A, b, eps):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    return gradient_descent(A, b, eps)


def conjugate_gradient_method_v2(A, b, eps):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    return gradient_descent(A, b, eps)

def conjugate_gradient_method_v3(A, b, eps):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    return gradient_descent(A, b, eps)
    
def gradient_descent(A, b, eps):
    m = len(A.T)
    x = np.zeros(shape=(m,1))
    i = 0
    imax = 1000
    r = b - A * x
    delta = r.T * r
    delta0 = delta
    while i < imax and delta > eps ** 2 * delta0:
        alpha = float(delta / (r.T * (A * r)))
        x = x + alpha * r
        r = b - A * x
        delta = r.T * r
        i += 1

    return x
