import numpy as np
import pandas as pd
import openpyxl

# Import and format data
df = pd.read_excel('KineticData2.xlsx', 'Sheet1').to_numpy()
t = df[:, 0]
C_A = df[:, 1]

# Initialize known parameters
C_A_initial = C_A[0]
C_B_initial = 0.053
h = 0.25  # delta time in minutes



