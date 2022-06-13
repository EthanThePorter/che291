import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.stats.distributions import t
from scipy import stats
import matplotlib.pyplot as plt
import openpyxl


class Lab5:

    def __init__(self):
        # Dict to save runtimes for each method to
        self.runtimes = {}

        # Task 1 FDF for loop calculation variables
        self.Rf2 = None
        self.CBf2 = None
        self.CAf2 = None
        self.trf2 = None

        # Task 1 FDF matrix calculation variables
        self.CBf1 = None
        self.CAf1 = None
        self.trf1 = None
        self.Rf1 = None

        # Task 1 CDF variables
        self.trc = None
        self.CAc = None
        self.CBc = None
        self.R_CDF = None

        # Task 1 BDF variables
        self.Rate_BDF = None

        # Task 2 variables
        self.tre = None
        self.CAe = None
        self.CBe = None
        self.Re = None
        self.nd = None
        self.nb = None

        # Get Data
        data = pd.read_excel('KineticDataFromChE101_vS22.xlsx', 'Sheet1')
        self.td = data.iloc[1:142, 0]
        self.CAd = data.iloc[1:142, 1]
        self.runtime = np.asarray(self.td, dtype=float)
        self.CA = np.asarray(self.CAd, dtype=float)

        # Initialize Known Parameters
        self.CAo = self.CA[0]
        self.CBo = 0.053
        self.CB = self.CA + self.CBo - self.CAo
        self.h = 0.25  # delta time in minutes
        self.n = len(self.runtime)  # n = 141


    def task1_FDF_forloop(self):
        # Calculate reaction rate using forward difference formula (EQN 6b) and ‘for’ loop
        self.trf2 = np.zeros(self.n - 4)
        self.CAf2 = np.zeros(self.n - 4)
        self.CBf2 = np.zeros(self.n - 4)
        self.Rf2 = np.zeros(self.n - 4)

        # Start Timer
        timer = time.time()
        # Calculate reaction rate using forward difference formula (EQN 6b) and array operations
        for i in range(self.n - 4):
            self.trf2[i] = self.runtime[i + 2]
            self.CAf2[i] = self.CA[i + 2]
            self.CBf2[i] = self.CBo - (self.CAo - self.CAf2[i])
            self.Rf2[i] = -(-3 * self.CA[i + 2] + 4 * self.CA[i + 3] - self.CA[i + 4]) / 2 / self.h
        # End timer and append value to main runtimes dict
        self.runtimes['1-FDF for-loop'] = time.time() - timer

    def task1_FDF(self):
        # Initialize Variables
        self.trf1 = self.runtime[2:-3]
        self.CAf1 = self.CA[2:-3]
        self.CBf1 = self.CA[2:-3] + self.CBo - self.CAo

        # Start Timer
        timer = time.time()
        # Calculate rate using Forward Difference Formula
        self.Rf1 = -(-3 * self.CA[2:-3] + 4 * self.CA[3:-2] - self.CA[4:-1]) / 2 / self.h
        # End timer and append value to main runtimes dict
        self.runtimes['1-FDF'] = time.time() - timer


    def task_1_CDF(self):
        # Initialize Variables
        self.trc = self.runtime[2:-3]
        self.CAc = self.CA[2:-3]
        self.CBc = self.CBo - (self.CAo - self.CAc)

        # Start Timer
        timer = time.time()
        # Calculate the rate using Central Difference Formula
        self.R_CDF = (self.CA[1:-4] - self.CA[3:-2]) / (2 * self.h)
        # End timer and append value to main runtimes dict
        self.runtimes['1-CDF'] = time.time() - timer


    def task_1_BDF(self):
        # Start Timer
        timer = time.time()
        # Calculate the rate using Backwards Difference Formula
        self.Rate_BDF = (-3 * self.CA[2:-3] + 4 * self.CA[1:-4] - self.CA[0:-5]) / 2 / self.h
        # End timer and append value to main runtimes dict
        self.runtimes['1-BDF'] = time.time() - timer

    def task_1_plot_results(self):
        fig, axes = plt.subplots(3)
        axes[0].plot(self.td[2:-3], self.Rf1)
        axes[0].legend('Forward')
        axes[1].plot(self.td[2:-3], self.R_CDF)
        axes[1].legend('Central')
        axes[2].plot(self.td[2:-3], self.Rate_BDF)
        axes[2].legend('Backward')
        plt.show()


    def task_2(self):
        # Calculate reaction rates using the smoothest method in Task 1 with less numerical errors
        trr = self.trc
        CAr = self.CAc
        CBr = self.CBc
        Rr = self.R_CDF

        j = 0
        while j <= (len(Rr) - 1):
            if Rr[j] <= 0:
                break
            else:
                j = j + 1

        ne = j - 1  # number of positive rate values
        # Re-assign new rate vectors for regression based on nr.
        self.tre = trr[:ne]
        self.CAe = CAr[:ne]
        self.CBe = CBr[:ne]
        self.Re = Rr[:ne]

        # Define the linearized data for multivariate linear regression
        y = np.log(self.Re)
        self.nd = len(y)
        self.nb = 3
        M = np.zeros((self.nd, self.nb))
        M[:, 0] = np.ones(self.nd)
        M[:, 1] = np.log(self.CAe)
        M[:, 2] = np.log(self.CBe)

        # Start timer
        timer = time.time()
        # Determine kinetic parameters with linear regression using Numpy’s linalg.lstsq
        beta, res, rank, S = np.linalg.lstsq(M, y, rcond=None)
        # End timer and append value to main runtimes dict
        self.runtimes['2-MLR'] = time.time() - timer

        # Get coefficient array and output
        k = np.exp(beta[0])
        print(f'\n\nTask 2: Coefficients\n{k, beta[1], beta[2]}')

        # Calculate SSE (EQN 3) , R2
        ypred = np.dot(M, beta)
        SSE = np.sum((y - ypred) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        R2 = 1 - SSE / SST
        print(f'\nTask 2: R_Squared\n{R2}')

        # Calculate 95% confidence intervals for the fitted model
        sigma2 = np.sum((y - np.dot(M, beta)) ** 2) / (self.nd - self.nb)  # sigma square
        var = sigma2 * np.linalg.inv(np.dot(M.T, M))  # covariance matrix
        se = np.sqrt(np.diag(var))  # standard error
        alpha = 0.05  # 100*(1 - alpha) confidence level
        tv = stats.t.ppf(1.0 - alpha / 2.0, self.nd - self.nb)  # student T multiplier
        CI = tv * se
        kCI = np.exp(CI[0])
        print(f'\nTask 2: Confidence Interval Values\n{kCI, CI[1], CI[2]}')


    def KineticEquation(self, tre, a0, a1, a2):
        R = a0 * self.CAe**a1 * self.CBe**a2
        return R


    def task_3(self):
        # Define an initial guess vector m0 for parameters and an upper bound/lower bound
        bnds = (0.001, 15)
        A0 = [6, 1, 1]

        # Start timer
        timer = time.time()
        # Use scipy’s curve_fit to perform nonlinear regression
        A, Acov = curve_fit(self.KineticEquation, self.tre, self.Re, A0, bounds=bnds)
        # End timer and append value to main runtimes dict
        self.runtimes['3-NLR'] = time.time() - timer
        print(f'\n\nTask 3: Coefficients\n{A}')


        # Calculate R_Squared
        rpred = self.KineticEquation(self.tre, A[0], A[1], A[2])
        rmean = np.mean(self.Re)
        SSE = np.sum((self.Re - rpred) ** 2)
        SST = np.sum((rpred - rmean) ** 2)
        R2 = 1 - SSE / SST
        print(f'\nTask 3: R-Squared\n{R2}')

        # Get 95% CI intervals
        se = np.sqrt(np.diag(Acov))  # standard error
        alpha = 0.05  # 100*(1 - alpha) confidence level
        tv = stats.t.ppf(1.0 - alpha / 2.0, self.nd - self.nb)  # student T multiplier
        CI = tv * se
        print(f'\nTask 3: Confidence Interval Values\n{CI}')


    def task_4(self):
        y = np.log(self.CA / (self.CBo - self.CAo + self.CA))

        trL = self.td[:-20]  # Remove data points at the end that deviate from the linear line
        yL = y[:-20]
        x_value = np.array([trL, np.ones(121)]).T

        # Start timer
        timer = time.time()
        # Get Rate constant
        func = np.linalg.lstsq(x_value, yL, rcond=None)
        slope1 = func[0][0]
        k4 = (-slope1) / (self.CBo - self.CAo)  # getting the k value from the slope of linalg.lstsq
        # End timer and append value to main runtimes dict
        self.runtimes['4-LS'] = time.time() - timer
        print(f'\n\nTask 4: k\n{k4}')


    def task_5(self):
        y = np.log(self.CA / (self.CBo - self.CAo + self.CA))
        trL = self.td[:-20]  # Remove data points at the end that deviate from the linear line
        yL = y[:-20]

        # Start Timer
        timer = time.time()
        # Calculate Rate
        slope, intercept, r_value, p_value, slope_SE = stats.linregress(trL, yL)
        # End timer and append value to main runtimes dict
        self.runtimes['5-LR'] = time.time() - timer
        k5 = (-slope) / (self.CBo - self.CAo)
        print(f'\n\nTask 5: k\n{k5}')


    @staticmethod
    def KineticEquations(C, t, m):

        # Define ODEs
        dCAdt = -m[0] * C[0] ** m[1] * C[1] ** m[2]
        dCBdt = -m[0] * C[0] ** m[1] * C[1] ** m[2]

        return dCAdt, dCBdt


    def PredictedCA(self, t, a0, a1, a2):

        # Converts 3 parameters into a list for inserting into kinetic equations function
        m = [a0, a1, a2]

        # Initial conditions
        y0 = self.CAo, self.CBo

        # Gets ODE solution
        sol = odeint(self.KineticEquations, y0, t, args=(m,))

        # Returns transpose of the solution's first column - This corresponds to the predicted CA data
        return np.transpose(sol)[0]


    def task_6(self):

        # Bounds and initial guesses for curvefitting function
        bounds = (0.001, 10)
        initial_parameter_guess = [6, 1, 1]

        # Start timer
        timer = time.time()
        # Find parameters for Kinetic Equations
        parameters, covariance = curve_fit(self.PredictedCA, self.runtime, self.CA, initial_parameter_guess, bounds=bounds)
        # End timer and append value to main runtimes dict
        self.runtimes['6-INM'] = time.time() - timer
        print(f'\n\nTask 6: Parameters\n{parameters}\n')

        # Get R squared for regression by comparing Integrated Numerical Method to Central difference formula
        R_predicted = self.KineticEquations([self.CA, self.CB], self.runtime, parameters)[0][2:-3]
        R_mean = np.mean(-self.R_CDF)
        SSE = np.sum((-self.R_CDF - R_predicted) ** 2)
        SST = np.sum((R_predicted - R_mean) ** 2)
        R2 = 1 - SSE / SST
        print(f'Task 6: R-Squared\n{R2}\n')

        # Get 95% CI for Integrated Numerical Method
        se = np.sqrt(np.diag(covariance))
        alpha = 0.05
        tv = stats.t.ppf(1.0 - alpha / 2.0, self.nd - self.nb)
        CI = tv * se
        print(f'Task 6: Confidence Interval Values\n{CI}\n\n')


# Initialize class and run all tasks
main = Lab5()
main.task1_FDF_forloop()
main.task1_FDF()
main.task_1_CDF()
main.task_1_BDF()
#main.task_1_plot_results()
main.task_2()
main.task_3()
main.task_4()
main.task_5()
main.task_6()


# Get dict key for fastest method
fastest = min(main.runtimes, key=main.runtimes.get)

# Convert runtimes into ms
for i in main.runtimes:
    main.runtimes[i] = round(main.runtimes[i] * 1000, 6)

# Get minimum runtime value
minimum = main.runtimes[fastest]

percent_slower = {}
for i in main.runtimes:

    percent_slower[i] = round((main.runtimes[i] / minimum - 1) * 100, 6)

print('\nMethod Runtime Comparisons:')
for i in main.runtimes:
    print(f'\n{i}:\n{main.runtimes[i]}ms, {percent_slower[i]}% Slower than {fastest}')
