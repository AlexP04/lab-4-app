# from solve import Solve
# from polynomial_builder import PolynomialBuilder
import itertools
from concurrent import futures
from time import time
from stqdm import stqdm

__author__ = 'nikiandr'

def print_stats(method_name, func_runtimes):
    if method_name not in func_runtimes:
        print("{!r} wasn't profiled, nothing to display.".format(method_name))
    else:
        runtimes = func_runtimes[method_name]
        total_runtime = sum(runtimes)
        average = total_runtime / len(runtimes)
        print('function: {!r}'.format(method_name))
        print(f'\trun times: {len(runtimes)}')
        # print('  total run time: {}'.format(total_runtime))
        print(f'\taverage run time: {average:.7f}')

def getError(params):
    params_new = params[-1].copy()
    Solver = params[-2]
    params_new['degrees'] = [*(params[:-2])]
    solver = Solver(params_new)
    func_runtimes = solver.prepare()
    normed_error = min(solver.norm_error)
    return (params_new['degrees'], normed_error, func_runtimes)

def getSolution(Solver, params, max_deg=15):
    if params['degrees'][0] == 0:
        x1_range = list(range(1, max_deg+1))
    else:
        x1_range = [params['degrees'][0]]
    
    if params['degrees'][1] == 0:
        x2_range = list(range(1, max_deg+1))
    else:
        x2_range = [params['degrees'][1]]
    
    if params['degrees'][2] == 0:
        x3_range = list(range(1, max_deg+1))
    else:
        x3_range = [params['degrees'][2]]

    ranges = list(itertools.product(x1_range, x2_range, x3_range, [Solver], [params]))
    # tick = time()
    if len(ranges) > 1:
        with futures.ThreadPoolExecutor() as pool:
            results = list(
                pool.map(getError, ranges)
            )

        results.sort(key=lambda t: t[1])
    else:
        results = [getError(ranges[0])]

    final_params = params.copy()
    final_params['degrees'] = results[0][0]
    solver = Solver(final_params)
    solver.prepare()
    tock = time()
    
    # print('\n--- BEGIN DEBUG INFO ---')

    # print(f'TOTAL RUNTIME: {tock-tick:.3f} sec\n\n')
    
    return solver