###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################
import numpy as np
import math

###############################################################################
# Basic Benchmark functions
###############################################################################

###############################################################################
# F1: Five-Uneven-Peak Trap
# Variable ranges: x in [0, 30
# No. of global peaks: 2
# No. of local peaks:  3.
def five_uneven_peak_trap(x=None):
    if x is None:
        return None
    x = x[:, 0]
    result = np.zeros_like(x)
    
    mask1 = (x >= 0) & (x < 2.50)
    mask2 = (x >= 2.5) & (x < 5)
    mask3 = (x >= 5.0) & (x < 7.5)
    mask4 = (x >= 7.5) & (x < 12.5)
    mask5 = (x >= 12.5) & (x < 17.5)
    mask6 = (x >= 17.5) & (x < 22.5)
    mask7 = (x >= 22.5) & (x < 27.5)
    mask8 = (x >= 27.5) & (x <= 30)
    
    result[mask1] = 80 * (2.5 - x[mask1])
    result[mask2] = 64 * (x[mask2] - 2.5)
    result[mask3] = 64 * (7.5 - x[mask3])
    result[mask4] = 28 * (x[mask4] - 7.5)
    result[mask5] = 28 * (17.5 - x[mask5])
    result[mask6] = 32 * (x[mask6] - 17.5)
    result[mask7] = 32 * (27.5 - x[mask7])
    result[mask8] = 80 * (x[mask8] - 27.5)
    
    return result


###############################################################################
# F2: Equal Maxima
# Variable ranges: x in [0, 1]
# No. of global peaks: 5
# No. of local peaks:  0.
def equal_maxima(x=None):

    if x is None:
        return None

    return np.sin(5.0 * np.pi * x[:, 0]) ** 6

###############################################################################
# F3: Uneven Decreasing Maxima
# Variable ranges: x in [0, 1]
# No. of global peaks: 1
# No. of local peaks:  4.
def uneven_decreasing_maxima(x=None):

    if x is None:
        return None

    return (
        np.exp(-2.0 * np.log(2) * ((x[:, 0] - 0.08) / 0.854) ** 2)
        * (np.sin(5 * np.pi * (x[:, 0] ** 0.75 - 0.05))) ** 6
    )


###############################################################################
# F4: Himmelblau
# Variable ranges: x, y in [-6, 6
# No. of global peaks: 4
# No. of local peaks:  0.
def himmelblau(x=None):

    if x is None:
        return None

    result = 200 - (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 - (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
    return result


###############################################################################
# F5: Six-Hump Camel Back
# Variable ranges: x in [-1.9, 1.9]; y in [-1.1, 1.1]
# No. of global peaks: 2
# No. of local peaks:  2.
def six_hump_camel_back(x=None):

    if x is None:
        return None

    x2 = x[:, 0] ** 2
    x4 = x[:, 0] ** 4
    y2 = x[:, 1] ** 2
    expr1 = (4.0 - 2.1 * x2 + x4 / 3.0) * x2
    expr2 = x[:, 0] * x[:, 1]
    expr3 = (4.0 * y2 - 4.0) * y2
    return -1.0 * (expr1 + expr2 + expr3)
    # result = (-4)*((4 - 2.1*(x[0]**2) + (x[0]**4)/3.0)*(x[0]**2) + x[0]*x[1] + (4*(x[1]**2) - 4)*(x[1]**2))
    # return result


###############################################################################
# F6: Shubert
# Variable ranges: x_i in  [-10, 10]^n, i=1,2,...,n
# No. of global peaks: n*3^n
# No. of local peaks: many
def shubert(x=None):

    if x is None:
        return None

    n, D = x.shape
    result = np.ones(n)
    soma = np.zeros((n, D))
    
    for j in range(1, 6):
        soma = soma + (j * np.cos((j + 1) * x + j))
    result = np.prod(soma, axis = 1)

    return -result


###############################################################################
# F7: Vincent
# Variable range: x_i in [0.25, 10]^n, i=1,2,...,n
# No. of global optima: 6^n
# No. of local optima:  0.
def vincent(x=None):

    if x is None:
        return None

    n, D = x.shape
    result = np.zeros(n)

    result = np.sum((np.sin(10 * np.log(x))) / D, axis = 1)
    return result


###############################################################################
# F8: Modified Rastrigin - All Global Optima
# Variable ranges: x_i in [0, 1]^n, i=1,2,...,n
# No. of global peaks: \prod_{i=1}^n k_i
# No. of local peaks:  0.
def modified_rastrigin_all(x=None):

    if x is None:
        return None

    n, D = x.shape
    
    if D == 2:
        k = [3, 4]
    elif D == 8:
        k = [1, 2, 1, 2, 1, 3, 1, 4]
    elif D == 16:
        k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]

    result = np.sum(10 + 9 * np.cos(2 * math.pi * np.array(k)[None, :] * x), axis=1)
    return -result
