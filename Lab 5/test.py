import numpy as np
import time
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from scipy import stats
import matplotlib.pyplot as plt
import openpyxl

# 3. Import and format data
data = pd.read_excel('KineticDataFromChE101_vS22.xlsx', 'Sheet1')
#runtime = data[0]
#Concentration_A = data[1]

print(data)


