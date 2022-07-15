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


def fugacity(mf, W):
    # Critical values:
    CPT = np.array([[48.2, 305.5, 0.099],  # Ethane
                    [42.0, 370.0, 0.153],  # Propane
                    [37.5, 425.2, 0.199],  # Butane
                    [33.75, 469.6, 0.254],  # Pentane
                    [30.32, 507.9, 0.300]])  # Hexane

    # --Task 1: Using the PR Equation
    # Equation 12
    a = 0.45724 * R ** 2 * CPT[:, 1] ** 2 / CPT[:, 0] * (
            1 + (0.37464 + 1.54226 * CPT[:, 2] - 0.26992 * CPT[:, 2] ** 2) * (
            1 - np.sqrt((T + 273.15) / CPT[:, 1]))) ** 2
    # Equation 13
    b = 0.07780 * R * CPT[:, 1] / CPT[:, 0]

    A = a * P / R ** 2 / (T + 273.15) ** 2
    B = b * P / R / (T + 273.15)

    Aprod = np.outer(A, A)
    # Equation 26
    Aij = (1 - 0.012) * Aprod ** 0.5
    Am = np.dot(mf, np.dot(Aij, mf))
    # Equation 25
    Bm = np.dot(mf, B)

    # Equation 14
    poly = [1, (Bm - 1), (Am - 2 * Bm - 3 * Bm ** 2), (Bm ** 3 + Bm ** 2 - Am * Bm)]
    Z = np.roots(poly)
    if W == 0:
        Zm = min(Z)
    else:
        Zm = max(Z)

    # Equation 27
    phi = np.exp(B / Bm * (Zm - 1) - np.log(Zm - Bm) - Am / 2 / np.sqrt(2) /
                 Bm * (2 * np.dot(mf, Aij) / Am - B / Bm) * np.log(
        (Zm + (1 + np.sqrt(2)) * Bm) / (Zm + (1 - np.sqrt(2)) * Bm)))

    return phi, Zm


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

mvL = np.zeros(Nn)
mvV = np.zeros(Nn)
mmL = np.zeros(Nn)
mmV = np.zeros(Nn)
Results1 = np.zeros((Nn, 2 * n + 2))

for i in range(Nn):

    P = Pop[i]

    mo = [0.3, 0.1, 0.2, 0.2, 0.2, 0.5, 0.3, 0.1, 0.05, 0.05, 0.5]  # Initial guessed values for unkowns

    m = optimize.fsolve(ProcessEquations, mo, args=(P,), xtol=1e-6)  # Solve the set of nonlinear equation.

    fp = np.zeros(2 * n + 1)
    x = m[0:n]  # Mole fractions of liquid phase.
    y = m[n:-1]  # Mole fraction of vapor phase.
    alpha = m[-1]  # Ratio of vaporization.

    phiL, ZL = fugacity(x, 0)
    phiV, ZV = fugacity(y, 1)
    mvL[i] = ZL * R * (T + 273.15) / P  # Molar volume (L/mol)
    mvV[i] = ZV * R * (T + 273.15) / P

    mm = np.array([30.04, 44.10, 58.12, 72.15, 86.18])  # molar mass vector
    mmL[i] = np.dot(x, mm)  # Average molar mass (g/mol) of liquid.
    mmV[i] = np.dot(y, mm)

    Results1[i, 0] = P
    Results1[i, 1:] = m

rhoL = mmL / mvL  # Density (g/cm3) of liquid.
rhoV = mmV / mvV

print("-="*60)
print('P', Pop)
print('rhoV', rhoV)
print('rhoL', rhoL)

plt.plot(Pop, rhoV, label='rhoV')
plt.title('Task 5 Results')
plt.xlabel('Pressure (atm)')
plt.ylabel('Density (g/cm3)')
plt.legend()
plt.show()

plt.plot(Pop, rhoL, label='rhoL')
plt.title('Task 5 Results')
plt.xlabel('Pressure (atm)')
plt.ylabel('Density (g/cm3)')
plt.legend()
plt.show()
