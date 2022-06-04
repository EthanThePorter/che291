import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from scipy import stats
import matplotlib.pyplot as plt
import openpyxl

# Import and format data
data = pd.read_excel('KineticDataFromChE101_vS22.xlsx', 'Sheet1')
runtime_raw = data.iloc[1:142, 0]
concentration_A_raw = data.iloc[1:142, 1]
runtime = np.asarray(runtime_raw, dtype=float)
concentration_A = np.asarray(concentration_A_raw, dtype=float)

# Initialize Known Parameters
concentration_A_initial = concentration_A[0]
concentration_B_initial = 0.053
h = 0.25  # delta time in minutes
n = len(runtime)  # n = 141

print(concentration_A)