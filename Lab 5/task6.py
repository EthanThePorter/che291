import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy import stats
import matplotlib.pyplot as plt
import openpyxl

# Import and format data
data = pd.read_excel('KineticDataFromChE101_vS22.xlsx', 'Sheet1')
runtime_raw = data.iloc[1:142, 0]
CA_raw = data.iloc[1:142, 1]
runtime = np.asarray(runtime_raw, dtype=float)
CA = np.asarray(CA_raw, dtype=float)

# Initialize Known Parameters
CAo = CA[0]
CBo = 0.053
h = 0.25
CB = CA + CBo - CAo


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

# Find parameters for Kinetic Equations and output result to terminal
parameters, covariance = curve_fit(PredictedCA, runtime, CA, initial_parameter_guess, bounds=bounds)
print(f'Parameters:\n{parameters}\n')

# Get statistics for regression by comparing Integrated Numerical Method to Central difference formula
R_predicted = KineticEquations([CA, CB], runtime, parameters)[0]
R_mean = np.mean(R_CDF)
#SSE = np.sum((Re - rpred)**2)

print(R_mean)




