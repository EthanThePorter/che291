import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd


def VLE(Pressure):
    # Gas Constant in atm.L/mol.K
    R = 0.08206
    # Number of components
    n = 2
    # Pressure in psi
    P_psi = Pressure
    # Convert pressure tp atm
    P = P_psi / 14.696

    # Number of points
    Nn = 101
    # Values for composition & temperature
    xb = np.zeros(Nn)
    yb = np.zeros(Nn)
    Tb = np.zeros(Nn)
    # Array to save results to
    Results1 = np.zeros((Nn, 2 * n + 2))

    def fugacity(mf, T, W):
        # Critical values:
        CPT = np.array([[42.0, 370.0, 0.153],  # Propane
                        [37.5, 425.2, 0.199]])  # Butane

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

    # Process Equations
    def VLE_Equations(m, P, x):

        fp = np.zeros(n + 1)

        y = [m[0], 1 - m[0]]
        T = m[1]
        alpha = m[-1]

        phiL, Zm = fugacity(x, T, 0)
        phiV, Zm = fugacity(x, T, 1)

        k = phiL / phiV

        z = [0.5, 0.5]
        fp[0:n] = y - k * x
        fp[n] = sum(z * (1 - k) / (1 + alpha * (k - 1)))
        return fp

    for i in range(Nn):
        xb[i] = i * 0.01
        x = [xb[i], 1 - xb[i]]
        z = [0.5, 0.5]

        yb0 = i * 0.01
        if Pressure == 350:
            T0 = 131 - (131 - 75) / Nn * i  # Initial temperature guess for 350psi
        elif Pressure == 200:
            T0 = 121 - (121 - 50) / Nn * i  # Initial temperature guess for 350psi
        else:
            T0 = 110 - (110 - 33) / Nn * i  # Initial temperature guess for 350psi
        alpha0 = i * 0.01

        # Initial Guess
        m0 = [yb0, T0, alpha0]

        # Solve equation
        m = optimize.fsolve(VLE_Equations, m0, args=(P, x), xtol=1e-6)
        yb[i] = m[0]
        Tb[i] = m[1]
        # alpha[i] = m{-1}

    return xb, yb, Tb


x_350, y_350, T_350 = VLE(350)
x_200, y_200, T_200 = VLE(200)
x_50, y_50, T_50 = VLE(50)

# Plot TXY Diagram
plt.plot(x_350, T_350 + 273.15, label='Dew Point Curve (350psi)')
plt.plot(y_350, T_350 + 273.15, label='Bubble Point Curve (350psi)')
plt.plot(x_200, T_200 + 273.15, label='Dew Point Curve (200psi)')
plt.plot(y_200, T_200 + 273.15, label='Bubble Point Curve (200psi)')
plt.plot(x_50, T_50 + 273.15, label='Dew Point Curve (50psi)')
plt.plot(y_50, T_50 + 273.15, label='Bubble Point Curve (50psi)')
plt.legend()
plt.xlabel('Mole Fraction Propane')
plt.ylabel('Temperature (K)')
plt.title(f'TXY for n-Butane-Propane System')
plt.xlim((0, 1))
plt.show()

# Plot XY Diagram
plt.plot(x_350, y_350, label='350psi')
plt.plot(x_200, y_200, label='200psi')
plt.plot(x_50, y_50, label='50psi')
plt.xlabel('Liquid Mole Fraction Propane')
plt.ylabel('Vapor Mole Fraction Propane')
plt.title(f'XY Diagram for n-Butane-Propane System ')
plt.legend()
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.show()

