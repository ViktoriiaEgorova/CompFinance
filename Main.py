import sys
import scipy
from scipy import interpolate
import math
import numpy as np
import matplotlib.pyplot as plt

import CrankNicolson
import lesson



if __name__ == '__main__':
    S0 = 90.0  # initial stock level
    K = 100.0  # strike price
    T = 1.0  # time-to-maturity
    r = 0.05  # short rate
    sigma = 0.20  # volatility
    delta = 0.0  # dividend yield
    N = 100

    opt_european = lesson.BlackScholes('P', S0, K, r, 0, sigma, T)
    opt_binomial_e = lesson.binomial_model_european(N, S0, sigma, r, K, T)[1][0][0]
    opt_binomial_a = lesson.binomial_model_american(N, S0, sigma, r, K, T)[1][0][0]
    #
    opt_explicit_e = lesson.finite_difference_explicit('put', S0, K, sigma, T, r, False)
    #
    opt_explicit = lesson.finite_difference_explicit('put', S0, K, sigma, T, r, True)
    #
    opt_implicit_e = lesson.finite_difference_implicit('put', S0, K, sigma, T, r, delta, False)
    #
    opt_implicit = lesson.finite_difference_implicit('put', S0, K, sigma, T, r, delta, True)
    opt_CN_LU_e = CrankNicolson.CrankNicolsonLU('put', r, K, S0, sigma, delta, T, False)
    opt_CN_SOR = CrankNicolson.CrankNicolsonSOR('put', r, K, S0, sigma, delta, T, True)


    print('Option price (analytic)               : ', round(opt_european, 3))
    print('Option price (binomial tree european) : ', round(opt_binomial_e, 3))
    print('Option price (binomial tree american) : ', round(opt_binomial_a, 3))
    #
    #print('Option price (explicit finite diff european)   : ', round(opt_explicit_e, 3))
    #
    print('Option price (explicit finite diff american)   : ', round(opt_explicit, 3))
    #
    #print('Option price (implicit finite diff eur)   : ', round(opt_implicit_e, 3))
    #
    print('Option price (implicit finite diff american)   : ', round(opt_implicit, 3))
    #print('Option price (Crank - Nicolson with LU european)   : ', round(opt_CN_LU_e, 3))
    print('Option price (Crank - Nicolson with SOR american)   : ', round(opt_CN_SOR, 3))











































