import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.stats.distributions import t
from scipy import stats
import matplotlib.pyplot as plt
import openpyxl


# 3. Import and format data
data = pd.read_excel('KineticDataFromChE101_vS22.xlsx', 'Sheet1')
td = data.iloc[1:142, 0]
CAd = data.iloc[1:142, 1]

runtime = np.asarray(td, dtype=float)
CA = np.asarray(CAd, dtype=float)



# 4. Initialize Known Parameters
CAo = CA[0]
CBo = 0.053
CB = CA + CBo - CAo
h = 0.25  # delta time in minutes
n = len(runtime)  # n = 141


print(CAo)


# 5. Calculate reaction rate using forward difference formula (EQN 6b) and ‘for’ loop
trf2 = np.zeros(n - 4)
CAf2 = np.zeros(n - 4)
CBf2 = np.zeros(n - 4)
Rf2 = np.zeros(n - 4)

start1 = time.perf_counter()
for i in range(n - 4):
    trf2[i] = runtime[i + 2]
    CAf2[i] = CA[i + 2]
    CBf2[i] = CBo - (CAo - CAf2[i])
    Rf2[i] = -(-3 * CA[i + 2] + 4 * CA[i + 3] - CA[i + 4]) / 2 / h

tf2 = (time.perf_counter() - start1) * 1000


# 6. Calculate reaction rate using forward difference formula (EQN 6b) and array operations
start1 = time.perf_counter()

trf1 = runtime[2:-3]
CAf1 = CA[2:-3]


CBf1 = CA[2:-3] + CBo - CAo

Rf1 = -(-3 * CA[2:-3] + 4 * CA[3:-2] - CA[4:-1]) / 2 / h

tf1 = (time.perf_counter() - start1) * 1000


# 7. Calculate reaction rate using central difference formula (EQN 8)
trc = runtime[2:-3]
CAc = CA[2:-3]
CBc = CBo - (CAo - CAc)

R_CDF = (CA[1:-4] - CA[3:-2]) / (2 * h)


# 8. Calculate reaction rate using backward difference (EQN 7b)
Rate_BDF = (-3 * CA[2:-3] + 4 * CA[1:-4] - CA[0:-5]) / 2 / h


# 9. Plot Results - CDF is smoothest
fig, axes = plt.subplots(3)
axes[0].plot(td[2:-3], Rf1)
axes[0].legend('Forward')
axes[1].plot(td[2:-3], R_CDF)
axes[1].legend('Central')
axes[2].plot(td[2:-3], Rate_BDF)
axes[2].legend('Backward')
#plt.show()


# Task 2
# 1. Calculate reaction rates using the smoothest method in Task 1 with less numerical errors
trr = trc
CAr = CAc
CBr = CBc
Rr = R_CDF

j = 0
while j <= (len(Rr) - 1):
    if Rr[j] <= 0:
        break
    else:
        j = j + 1

ne = j - 1  # number of positive rate values
print(ne)
# Re-assign new rate vectors for regression based on nr.
tre = trr[:ne]
CAe = CAr[:ne]
CBe = CBr[:ne]
Re = Rr[:ne]


# 2. Define the linearized data for multivariate linear regression
y = np.log(Re)
nd = len(y)
nb = 3
M = np.zeros((nd, nb))
M[:, 0] = np.ones(nd)
M[:, 1] = np.log(CAe)
M[:, 2] = np.log(CBe)


# 3. Determine kinetic parameters with linear regression using Numpy’s linalg.lstsq
beta, res, rank, S = np.linalg.lstsq(M, y, rcond=None)

print(beta)


k = np.exp(beta[0])
beta[0] = k
print('\nBeta Array Value [k a B]')
print(k, beta[1], beta[2])


# 4. Calculate SSE (EQN 3) , R2
ypred = np.dot(M, beta)
SSE = np.sum((y-ypred)**2)
SST = np.sum((y - np.mean(y))**2)
R2 = 1 - SSE / SST

print('\nR-Squared Value')
print(R2)


# 5. Calculate 95% confidence intervals for the fitted model
sigma2 = np.sum((y - np.dot(M, beta))**2) / (nd - nb)  # sigma square
var=sigma2 * np.linalg.inv(np.dot(M.T, M))  # covariance matrix
se = np.sqrt(np.diag(var))  # standard error
alpha = 0.05  # 100*(1 - alpha) confidence level
tv = stats.t.ppf(1.0-alpha/2.0, nd-nb)  # student T multiplier
CI = tv*se
kCI = np.exp(CI[0])

print('\nTask 2: Confidence Interval Values')
print(kCI, CI[1], CI[2])


# TASK 3
# 1. Define Kinetics Equation
def KineticEquation(tre, a0, a1, a2):

    R = a0 * CAe**a1 * CBe**a2
    return R


# 2. Define an initial guess vector m0 for parameters and an upper bound/lower bound
bnds = (0.001, 15)
A0 = [6, 1, 1]


# 3. Use scipy’s curve_fit to perform nonlinear regression
A, Acov = curve_fit(KineticEquation, tre, Re, A0, bounds=bnds)


# 4. Calculate SSE, R2, and 95% CIs values using the same method as Task 2
m = [k, beta[1], beta[2]]
rpred = KineticEquation(tre, A[0], A[1], A[2])
rmean = np.mean(Re)
SSE = np.sum((Re - rpred)**2)
SST = np.sum((rpred - rmean)**2)
R2 = 1 - SSE / SST

print(f'R_mean:{rmean}')

se = np.sqrt(np.diag(Acov))  # standard error
alpha = 0.05  # 100*(1 - alpha) confidence level
tv = stats.t.ppf(1.0-alpha/2.0, nd-nb)  # student T multiplier
CI = tv*se


print('\nTask 3: Coefficients')
print(A)

print('\nTask 3: R-Squared Value for Non-Linear Regression')
print(R2)

print('\nTask 3: Confidence Interval Values')
print(CI)


# Task 6
def KineticEquations(C, t, m):

    # Define ODEs
    dCAdt = -m[0] * C[0]**m[1] * C[1]**m[2]
    dCBdt = -m[0] * C[0]**m[1] * C[1]**m[2]

    return dCAdt, dCBdt


def PredictedCA(t, a0, a1, a2):

    # Converts 3 parameters into a list for inserting into kinetic equations function
    m = [a0, a1, a2]

    # Initial conditions
    y0 = CAo, CBo

    # Gets ODE solution
    sol = odeint(KineticEquations, y0, t, args=(m,))

    # Returns transpose of the solution's first column - This corresponds to the predicted CA data
    return np.transpose(sol)[0]


# Bounds and initial guesses for curvefitting function
bounds = (0.001, 10)
initial_parameter_guess = [6, 1, 1]

# Find parameters for Kinetic Equations
parameters, covariance = curve_fit(PredictedCA, runtime, CA, initial_parameter_guess, bounds=bounds)
print(f'\n\nTask 6: Parameters\n{parameters}\n')

# Get R squared for regression by comparing Integrated Numerical Method to Central difference formula
R_predicted = KineticEquations([CA, CB], runtime, parameters)[0][2:-3]
R_mean = np.mean(-R_CDF)
SSE = np.sum((-R_CDF - R_predicted) ** 2)
SST = np.sum((R_predicted - R_mean) ** 2)
R2 = 1 - SSE / SST
print(f'Task 6: R-Squared\n{R2}\n')

# Get 95% CI for Integrated Numerical Method
se = np.sqrt(np.diag(covariance))
alpha = 0.05
tv = stats.t.ppf(1.0-alpha/2.0, nd-nb)
CI = tv*se
print(f'Task 6: Confidence Interval Values\n{CI}\n')
