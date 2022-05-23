import numpy as np
import time
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from scipy import stats
import matplotlib.pyplot as plt
import openpyxl

# 3. Import and format data
data = pd.read_excel('KineticDataFromChE101_vS22.xlsx','Sheet1')
td = data.iloc[1:142, 0]
CAd = data.iloc[1:142, 1]

tr = np.asarray(td, dtype=float)
CA = np.asarray(CAd, dtype=float)


# 4. Initialize Known Parameters
CAo = CA[0]
CBo = 0.053
h = 0.25  # delta time in minutes
n = len(tr)  # n= 141


# 5. Calculate reaction rate using forward difference formula (EQN 6b) and ‘for’ loop
trf2 = np.zeros(n - 4)
CAf2 = np.zeros(n - 4)
CBf2 = np.zeros(n - 4)
Rf2 = np.zeros(n - 4)

start1 = time.perf_counter()
for i in range(n - 4):
    trf2[i] = tr[i + 2]
    CAf2[i] = CA[i + 2]
    CBf2[i] = CBo - (CAo - CAf2[i])
    Rf2[i] = -(-3 * CA[i + 2] + 4 * CA[i + 3] - CA[i + 4]) / 2 / h

tf2 = (time.perf_counter() - start1) * 1000
print(tf2)


# 6. Calculate reaction rate using forward difference formula (EQN 6b) and array operations
start1 = time.perf_counter()

trf1 = tr[2:-3]
CAf1 = CA[2:-3]

CBf1 = CAf1 + CBo - CAo

Rf1 = -(-3 * CAf1 + 4 * CA[3:-2] - CA[4:-1]) / 2 / h

tf1 = (time.perf_counter()-start1)*1000
print(tf1)


# 7. Calculate reaction rate using central difference formula (EQN 8)
Rate_CDF = (CA[:-2] - CA[2:]) / 2 / h

print(Rf1[:5])
print(Rate_CDF[:5])









