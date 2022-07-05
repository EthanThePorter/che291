# Experiemnt6_Task1_template

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

# gas constant.
R = 0.08206

# Feed temperature?
# Temperature in oC
T = 59
# Pressure in atm.
Pop = np.arange(10.0, 32.0, 2.0)

# Feed molar flow rate, mol/s.
F = 1
# Feed composition in mole fraction.
z = np.array([0.3977, 0.2926, 0.1997, 0.0731, 0.0369])

Nn = len(Pop)
n = len(z)



def fugacity(mf,W):

    # Critical values:
    CPT=np.array([[48.2,  305.5,  0.099],              # Ethane
                  [42.0,  370.0,  0.153],              # Propane
                  [37.5,  425.2,  0.199],              # Butane
                  [33.75, 469.6,  0.254],              # Pentane
                  [30.32, 507.9,  0.300]])             # Hexane

    #--Task 1: Using the PR Equation
    # Equation 12
    a=0.45724*R**2*CPT[:,1]**2/CPT[:,0]*(1+(0.37464+1.54226*CPT[:,2]-0.26992*CPT[:,2]**2)*(1-T/CPT[:,1]))**2
    # Equation 13
    b=0.07780*R*CPT[:, 1]/CPT[:,0]

    A=a*P/R**2/(T+273.15)**2
    B=b*P/R/(T+273.15)

    Aprod = np.outer(A,A)
    # Equation 26
    Aij= (1-0.012) * Aprod ** 0.5
    Am=np.dot(mf,np.dot(Aij,mf))
    # Equation 25
    Bm=np.dot(mf,B)

    # Equation 14
    poly=[1, (B - 1), (A - 2*B - 3*B**2), (B**3 + B**2 - A * B)]
    Z=np.roots(poly)
    if W==0:
       Zm=min(Z)
    else:
       Zm=max(Z)

    # Equation 27
    phi=np.exp(B/Bm*(Zm-1)-np.log(Zm-Bm)-
        Am/2/np.sqrt(2)

    return phi,Zm


def ProcessEquations(m, P):
    fp = np.zeros(2 * n + 1)
    x = m[0:n]  # Mole fractions of liquid phase.
    y = m[n:-1]  # Mole fraction of vapor phase.
    alpha = m[-1]  # Ratio of vaporization.

    phiL, Zm = fugacity(x, 0)  # Fugasity coefficient of liquid
    phiV, Zm = fugacity(y, 1)  # Fugasity coefficient of vapor

    k = phiL / phiV  # Equilibrium constant
    print(k)

    fp[0:n] = y - k * x
    fp[n:2 * n] = x - z / (1 + alpha * (k - 1))
    fp[2 * n] = sum(z * (1 - k) / (1 + alpha * (k - 1)))  # Equation 8
    return (fp)


Results1 = np.zeros((Nn, 2 * n + 2))

for i in range(Nn):
    P = Pop[i]

    mo = [0.3, 0.1, 0.2, 0.2, 0.2, 0.5, 0.3, 0.1, 0.05, 0.05, 0.5]  # Initial guessed values for unkowns

    m = optimize.fsolve(ProcessEquations, mo, args=(P), xtol=1e-6)  # Solve the set of nonlinear equation.
    Results1[i, 0] = P
    Results1[i, 1:] = m

    x = m[0:n]  # Mole fractions of liquid phase.
    y = m[n:2 * n]  # Mole fraction of vapor phase.
    alpha = m[-1]  # Ratio of vaporization.

Results1 = np.round(Results1, 4)
aa = Results1.shape

index = np.arange(1, Nn + 1)
pd.set_option("display.max_rows", None, "display.max_columns", None)  # option to display all rows and columns
columns = ['Pressure', 'xC2H6', 'xC3H8', 'xC4H10', 'xC5H12', 'xC6H14', 'yC2H6', 'yC3H8', 'yC4H10', 'yC5H12', 'yC6H14',
           'alpha']
Results11 = pd.DataFrame(Results1, index, columns)
ab = Results11.shape

Results11.to_excel('ChE291_E6_Task1.xlsx')
