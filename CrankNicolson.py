import sys

import scipy
from scipy import interpolate

import math
import numpy as np
import matplotlib.pyplot as plt
import lesson

def CrankNicolsonSOR(payout, r, K, S0, sigma, delta, T, american = True):
    S_min = 0.0;
    S_max = 150.0;
    L = S_max - S_min;
    N = 1002;  # Number of time steps
    dt = float(T) / float(N);  # time step size
    I = 100;  # Number of space steps
    dS = float(L) / float(I);  # space step size

    omega = 1.2
    EPS = 0.00001
    MAXITER = 100
    error = 1e6

    ####################
    # PREPARATION

    Uk = np.zeros((I + 1))  # [0, 1, ..., I]
    Uk_next = np.zeros((I + 1))
    S = np.zeros((I + 1))

    # Initial values - vector at time zero is fully known
    for i in range(0, I + 1):
        S[i] = i * dS
        if payout == 'call':
            Uk[i] = np.maximum(S[i] - K, 0)
        else:
            Uk[i] = np.maximum(K - S[i], 0)
    Payoff = Uk

    ####################
    # CRANK - NICOLSON

    # Cycle on time starting from known initial values
    for k in range(1, N - 1):

        sys.stdout.write("\r" + "Now running time step nr: " + str(k) + "/" + str(N - 2))
        sys.stdout.flush()
        t = k * dt
        m = 0
        # initially U at the current step is set equal to U at the previous step
        Uk_next[1:I - 1] = Uk[1:I - 1]
        # 1
        # Boundary conditions
        if payout == 'call':
            Uk_next[0] = 0
            Uk_next[I] = dS * I * math.exp(-delta * t) - K * math.exp(-r * t)
        else:
            Uk_next[0] = K * math.exp(-r * t)
            Uk_next[I] = 0
        # Now we completely know current vector of values Uk_next
        # 2
        # Matrices A, b = RHS
        RHS = np.zeros((I - 1))
        mA = np.zeros((I - 1))
        mB = np.zeros((I - 1))
        mC = np.zeros((I - 1))
       # A = np.zeros((I - 1, I - 1))
        for i in range(1, I):
            a = 0.25 * dt * i * (sigma * sigma * i - r)
            b = - 0.5 * dt * (r + sigma * sigma * i * i)
            c = 0.25 * dt * i * (sigma * sigma * i + r)
            A_right = a
            B_right = 1 + b
            C_right = c
            A_left = -a
            B_left = 1 - b
            C_left = -c
            if i == 1:
                RHS[i - 1] = A_right * Uk[i - 1] + B_right * Uk[i] + C_right * Uk[i + 1] - A_left * Uk_next[0]
                #A[i - 1][i - 1] = B_left
                #A[i - 1][i] = C_left
                mB[i - 1] = B_left
                mC[i - 1] = C_left
            elif i == I - 1:
                RHS[i - 1] = A_right * Uk[i - 1] + B_right * Uk[i] + C_right * Uk[i + 1] - C_left * Uk_next[I]
                mB[i - 1] = B_left
                mA[i - 1] = A_left
            else:
                RHS[i - 1] = A_right * Uk[i - 1] + B_right * Uk[i] + C_right * Uk[i + 1]
                #A[i - 1][i - 1] = B_left
                #A[i - 1][i - 2] = A_left
                #A[i - 1][i] = C_left
                mB[i - 1] = B_left
                mA[i - 1] = A_left
                mC[i - 1] = C_left
        # Now we have equation A*U_{k+1} = RHS

        # 3
        # SOR
        while (error > EPS) and (m < MAXITER):
            error = 0
            for i in range(1, I):
                y = (RHS[i - 1] - mA[i - 1] * Uk_next[i - 1] - mC[i - 1] * Uk_next[i + 1]) / mB[i - 1]
                diff = y - Uk_next[i]
                error = error + diff * diff
                Uk_next[i] = Uk_next[i] + omega * diff
                # early exercise opportunity
                if american:
                    Uk_next[i] = max(Uk_next[i], Payoff[i])
            m = m + 1
        #
        Uk[1:I - 1] = Uk_next[1:I - 1]

    ####################
    # OPTION PRICE

    f = scipy.interpolate.interp1d(S, Uk_next)
    opt = f(S0).item(0)
    print('\r')

    return opt





def CrankNicolsonLU(payout, r, K, S0, sigma, delta, T, american = True):

    #print('American option price with Crank-Nicolson method')

    #payout = 'put'
    S_min = 0.0
    S_max = 150.0
    L = S_max - S_min
    N = 1002  # Number of time steps
    dt = float(T) / float(N)  # time step size #k
    I = 100  # Number of space steps
    dS = float(L) / float(I)  # space step size    #h

    ####################
    # PREPARATION

    U0 = np.zeros((I + 1))  # [0, 1, ..., I]
    Uk = np.zeros((I + 1))
    Uk_next = np.zeros((I + 1))
    S = np.zeros((I + 1))
    Payoff = np.zeros((I + 1))
    M = np.zeros((I - 1, I + 1))
    M1 = np.zeros((I - 1, I + 1))
    A = np.zeros(I - 1)
    B = np.zeros(I - 1)
    C = np.zeros(I - 1)
    R = np.zeros(I - 1)
    ML = np.zeros((I - 1, I - 1))
    RHS = np.zeros(I - 1)
    X = np.zeros(I - 1)

    # Initial values - vector at time zero is fully known
    for i in range(0, I + 1):
        S[i] = i * dS
        if payout == 'call':
            U0[i] = np.maximum(S[i] - K, 0)
        else:
            U0[i] = np.maximum(K - S[i], 0)
    Payoff = U0
    Uk = U0

    ####################
    # CRANK - NICOLSON

    # Cycle on time starting from known initial values
    for k in range(1, N - 1):

        sys.stdout.write("\r" + "Now running time step nr: " + str(k) + "/" + str(N - 2))
        sys.stdout.flush()
        t = k * dt
        m = 0
        # 1
        # Boundary conditions
        if payout == 'call':
            Uk_next[0] = 0
            Uk_next[I] = dS * I * math.exp(-delta * t) - K * math.exp(-r * t)
        else:
            Uk_next[0] = K * math.exp(-r * t)
            Uk_next[I] = 0
        # Now we completely know current vector of values Uk
        # 2
        # Matrices of coefficients M, M1 + A, B, C
        for i in range(1, I):
            a = 0.25 * dt * i * (sigma * sigma * i - r)
            b = - 0.5 * dt * (r + sigma * sigma * i * i)
            c = 0.25 * dt * i * (sigma * sigma * i + r)
            # rigth
            M1[i - 1][i - 1] = a
            M1[i - 1][i] = 1 + b
            M1[i - 1][i + 1] = c
            # left
            M[i - 1][i - 1] = -a
            M[i - 1][i] = 1 - b
            M[i - 1][i + 1] = -c
            # columns
            A[i - 1] = -a
            B[i - 1] = 1 - b
            C[i - 1] = -c
        # Vector G
        G = np.dot(M1, Uk)
        # 3
        # Now we have equation M*U_{k+1} = G
        # In order to implement boundary conditions, we need to rearrange this equation ML*U_{k+1} + R = G
        R[0] = M[0][0] * Uk_next[0]
        R[-1] = M[-1][-1] * Uk_next[-1]
        ML = M[:, 1:-1]
        RHS = G - R
        # Now we have equation ML*U_{k+1} = RHS
        # 3
        # SOR
        # Python solver
        X = np.linalg.solve(ML, RHS)
        Uk_next[1:-1] = X
        if american:
            for i in range(1, I):
                Uk_next[i] = max(Uk_next[i], Payoff[i])
        # 4
        # Transer to the next step
        Uk = Uk_next

    ####################
    # OPTION PRICE

    f = scipy.interpolate.interp1d(S, Uk_next)
    opt = f(S0).item(0)
    print('\r')
    return opt





if __name__ == '__main__':
    S0 = 90.0  # initial stock level
    K = 100.0  # strike price
    T = 1.0  # time-to-maturity
    r = 0.05  # short rate
    sigma = 0.20  # volatility
    delta = 0.0  # dividend yield


    opt_CN_SOR = CrankNicolsonSOR('put', r, K, S0, sigma, delta, T, True)
    print('Option price (Crank - Nicolson american)   : ', round(opt_CN_SOR, 3))





























