from scipy import integrate
import numpy as np
#import sympy as smp
import pandas as pd
import matplotlib.pyplot as plt
import json

# Data filtering

with open("PythonTest/IntegrationCalculatorWebsite/data_under_3.json", "r") as file:
    data = json.load(file)
dataframe = pd.DataFrame(data)[["position_meters", "velocity_ms"]].values

class BandoFtL:

    def __init__(self, dataframe):
        """Class Manager for the BandoFtL traffic model. 
        Takes in a Panda's Dataframe that has already been cleaned. Has specific Alpha, Beta, and Car length Values that can be adjusted manually."""
        self.car_length = 4.5
        self.Vmax = 3 # Changed from 10 -> 3 to ensure the car stays within the scale of the generated data
        self.alpha = 0.5
        self.beta = 20
        self.ds = 2.5

        # Initial positions
        self.initial_positions = dataframe[:, 0].copy()
        # Initial velocities
        self.initial_velocities = dataframe[:, 1].copy()

        # Length of the data (step count)
        self.data_length = len(dataframe)

    def velocity_func(self, headway):
        """Takes a headay from the car and returns the optimal velocty for that position"""
        velocity = self.Vmax * (np.tanh(headway - self.ds) + np.tanh(self.car_length + self.ds)) / (1.0 + np.tanh(self.car_length + self.ds))
        return velocity

    def Acc(self, x, xl, v, vl):
        headway = xl - x - self.car_length
        # Check that the headway is greater than 0, the cars did not crash
        if headway > 0:
            # Defines the Bando Term
            bandoterm = self.alpha * (self.velocity_func(headway) - v)
            # Defines the FtL term
            ftlterm = self.beta * ((vl - v) / (xl - x - self.car_length)**2)
            return bandoterm + ftlterm  
        else: 
            return 0 # Returns 0 to ensure that the script doesn't crash when the headway does not compute


    def system(self, t, z):

        x = float(z[0])
        v = float(z[1])

        # New Leading terms
        # Interpelate the leading x and v from the intial position and velocity
        x_lead = np.interp(t, range(0, self.data_length), self.initial_positions) # WORKS 
        v_lead = np.interp(t, range(0, self.data_length), self.initial_velocities) # WORKS

        # Equations to be used by the solver
        dxdt = v
        dvdt = self.Acc(x, x_lead, v, v_lead)

        return [float(dxdt), float(dvdt)]


    def solve(self, eval_time):
        # Create the time threshold to run, can't have more steps than the lenght of the data set
        time = np.linspace(0, eval_time, self.data_length)

        # Create the intial conditions and start the following car at a distance of 10 behind the leading car
        y_0 = np.array([self.initial_positions[0] - 10, self.initial_velocities[0]])

        # Solve the system of the differential equaitons
        sol = integrate.solve_ivp(fun=self.system, t_span=(0, eval_time), y0=y_0, t_eval=time)

        return sol


# Create the Time frame
time = len(dataframe)   

manager = BandoFtL(dataframe)

# Solve the system
values = manager.solve(120)

vel_pos = False  # True is velocity, False is position

if vel_pos:
    # Plot the velocity of the leading car
    plt.plot(values.t, manager.initial_velocities[:len(values.t)], color='red', linestyle='--')
   
    # Plots the velocity of the following car
    plt.plot(values.t, values.y[1], color='blue')
else:
    #Plot the position of the leading car
    plt.plot(values.t, manager.initial_positions[:len(values.t)], color="red", linestyle='--')

    # Plot the position of the following car
    plt.plot(values.t, values.y[0], color='blue')

plt.show()
