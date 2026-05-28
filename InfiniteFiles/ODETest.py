from scipy import integrate
import numpy as np
import sympy as smp
import pandas as pd
import matplotlib.pyplot as plt

# Data filtering
dataframe = pd.read_csv("IntegrationCalculatorWebsite/data.csv")

car_data = dataframe["Velocity"].to_numpy()

# Constants
X_0 = 1
V_0 = 0
Xl_0 = 1
Vl_0 = 0
# TIME HORIZON MUST BE THE SAME SIZE AS THE DATA THAT YOU ARE PASSING IN
TIME_HORIZON = 31
ALPHA = 1
BETA = 1
vmax = 1
ds = 2.5
length = 4.5
STEP = 101

# Time values
time = np.linspace(0, TIME_HORIZON, STEP)

# Functions and the Derivtaives
def dvdt(t, x_t, xl_t, v_t, vl_t, alpha, beta, length, ds, vmax):
    term1 = alpha * vmax * (np.tanh((xl_t-x_t - length)) + np.tanh(length + ds)/(1 + np.tanh(length +ds))) - v_t
    term2 = beta * ((vl_t - v_t)/(xl_t - x_t - length)**2)
    return term1 + term2

def dSdx(x, S):
    # Constants:
    alpha = 1
    beta = 1
    length = 4.5
    ds = 2.5

    x_t, xl_t, v_t, vl_t = S
    term1 = alpha * vmax * (np.tanh((xl_t-x_t - length)) + np.tanh(length + ds)/(1 + np.tanh(length +ds))) - v_t
    term2 = beta * ((vl_t - v_t)/(xl_t - x_t - length)**2)
    return [np.linspace(0, TIME_HORIZON, STEP), v_t, term1 + term2]

S_0 = (0, 0, 0, 0)

#solved = solve_ivp(dSdx, y0=S_0, t=time, tfirst=True)

# Integrate the data set to obtain the position data:
def integrate_vel_data(vel_data, time_horizon):
    print(np.linspace(0, time_horizon, vel_data.size))
    print(vel_data.size)
    if time_horizon >= vel_data.size:
        pos_data = integrate.simpson(vel_data, np.linspace(0, time_horizon, vel_data.size))
        return pos_data
    else:
        print("Error, Time Horizon Given does not match!")
        return None
    
# Find the delta X for each position and save it to a list to be used later for the ACC definition
# Returns a list of values that contani the integrated velocity i the previous stpes integrated velocity: Delta x
def delta_x_integrated(vel_data, time_horizon):
    steps = np.linspace(0, time_horizon - 1, vel_data.size)
    sum = 0
    deltaxlist = np.array([])
    if time_horizon >= vel_data.size:
        for i in range(vel_data.size -2):
            prevx = integrate.simpson(vel_data[0:i+1])
            nextx = integrate.simpson(vel_data[0:i+2])
            delx = float(nextx - prevx)
            deltaxlist = np.append(deltaxlist, delx) # type: ignore
        return deltaxlist
    else:
        print("Error, Time_horion does not match!")
        return None


delta_x = delta_x_integrated(car_data, TIME_HORIZON)

# Acc function defined in terms of the delta x and velocity data
def acc(vel_data, time_horizon, alpha, beta, delta_xi, length, ds):
    accc = np.array([])
    if time_horizon >= vel_data.size:
        print(vel_data.size -2 )
        print(delta_xi.size)
        for i in range(vel_data.size -2):
            term1 = alpha * (vmax * (np.tanh(delta_xi[i] - length) + np.tanh(length + ds))/(1 + np.tanh(length + ds)) - vel_data[i])
            # ADD TERM 2 FROM THE FtL MODEL LATER
            accc = np.append(accc, term1)
        return accc

# integrate the Acc function to return the vlaue of the velocty of the following car
def integrate_acc(vel_data, time_horizon, alpha, beta, delta_xi, length, ds):
    deltaalist = np.array([])
    accel = acc(vel_data, time_horizon, alpha, beta, delta_xi, length, ds)
    if time_horizon >= vel_data.size:
        for i in range(vel_data.size -2):
            preva = integrate.simpson(accel[0:i+1])
            nexta = integrate.simpson(accel[0:i+2])
            dela = float(nexta - preva)
            deltaalist = np.append(deltaalist, dela) # type: ignore
        return deltaalist
    else:
        print("Error, Time_horion does not match!")
        return None

plt.plot(car_data)
plt.plot(delta_x)
print(delta_x)
print(type(delta_x))

acceleration = acc(car_data, TIME_HORIZON, ALPHA, BETA, delta_x, length, ds)

plt.plot(acceleration)
print(acceleration)

integrated = integrate_acc(car_data, TIME_HORIZON, ALPHA, BETA, delta_x, length, ds)
print(integrated)
plt.plot(integrated)
plt.show()

#print(integrate_vel_data(car_data, TIME_HORIZON))

#print(solved)
