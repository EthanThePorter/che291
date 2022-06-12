import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy import stats
import matplotlib.pyplot as plt
import openpyxl


# # DE
# def f(y, t):
#     k = -2
#     dydt = -k * y
#     return dydt
#
#
# # IC
# y0 = 1
#
# # Range to calculate over
# t = np.linspace(0, 2)
#
# # Find DE solution
# y = odeint(f, y0, t)
#
#
# plt.plot(t, y)
# plt.show()

x = [1, 2, 3]
y = [1, 2, 3]

print(x + y)