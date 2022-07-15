# Experiemnt6_Task1_template

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

# gas constant.
R = 0.08206

# Pressure in atm.
Pop = np.arange(10.0, 32.0, 2.0)

# Feed molar flow rate, mol/s.
F = 1
# Feed composition in mole fraction.
z = np.array([0.3977, 0.2926, 0.1997, 0.0731, 0.0369])

Nn = len(Pop)
n = len(z)


def fugacity(mf, T, W):
    # Critical values:
    CPT = np.array([[48.2, 305.5, 0.099],  # Ethane
                    [42.0, 370.0, 0.153],  # Propane
                    [37.5, 425.2, 0.199],  # Butane
                    [33.75, 469.6, 0.254],  # Pentane
                    [30.32, 507.9, 0.300]])  # Hexane

    # --Task 1: Using the PR Equation
    # Equation 33
    a = 0.45724 * R ** 2 * CPT[:, 1] ** 2 / CPT[:, 0] * (
            1 + (0.37464 + 1.54226 * CPT[:, 2] - 0.26992 * CPT[:, 2] ** 2) * (
                1 - np.sqrt((T + 273.15) / CPT[:, 1]))) ** 2
    # Equation 32
    b = 0.07780 * R * CPT[:, 1] / CPT[:, 0]

    A = a * P / R ** 2 / (T + 273.15) ** 2
    B = b * P / R / (T + 273.15)

    Aprod = np.outer(A, A)
    # Equation 26
    Aij = (1 - 0.012) * Aprod ** 0.5
    Am = np.dot(mf, np.dot(Aij, mf))
    # Equation 25
    Bm = np.dot(mf, B)

    # Equation 34
    poly = [1, (Bm - 1), (Am - 2 * Bm - 3 * Bm ** 2), (Bm ** 3 + Bm ** 2 - Am * Bm)]
    Z = np.roots(poly)
    if W == 0:
        Zm = min(Z)
    else:
        Zm = max(Z)

    # Equation 36
    phi = np.exp(B / Bm * (Zm - 1) - np.log(Zm - Bm) - Am / 2 / np.sqrt(2) /
                 Bm * (2 * np.dot(mf, Aij) / Am - B / Bm) * np.log(
        (Zm + (1 + np.sqrt(2)) * Bm) / (Zm + (1 - np.sqrt(2)) * Bm)))

    return phi, Zm


def BubblePointEquations(m):
    fp = np.zeros(n + 1)
    x = z  # Mole fractions of liquid phase.
    y = m[0:n]  # Mole fraction of vapor phase.
    T = m[-1]  # Ratio of vaporization.

    phiL, Zm = fugacity(x, T, 0)
    phiV, Zm = fugacity(y, T, 1)

    # Equation 30
    k = phiL / phiV
    # Equation 29
    fp[0:n] = y - k * x
    fp[n] = sum(z / k) - 1
    return fp


Results1 = np.zeros((Nn, 2 * n + 2))

Tob = np.zeros(Nn)
P = np.zeros(Nn)
Tbp = np.zeros(Nn)

for i in range(Nn):
    P = Pop[i]
    Tob[i] = -20 + i * 7  # Guessed bubble point in Celsius
    mob = [0.65, 0.15, 0.1, 0.05, 0.05, Tob[i]]
    mb = optimize.fsolve(BubblePointEquations, mob, xtol=1e-6)

    Tbp[i] = mb[-1]

print(Tbp)
print(P)


