import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime

import time

class BandoFtl:

    def __init__(self, parameters, time_horizon):
        """Class Manager for the BandoFtl model"""
        self.params = parameters
        self.T = time_horizon
        self.dt = 0.01
        self.num_vehicles = 22
        self.vehicle_length = params['vehicle_length']

        self.wave_started = []
        self.wave_ended = []
        self.wave_threshold = 0.2
        self.wave_placeholder = 0

        # Set the interval for the distractions
        self.lower_bound_interval = params['lower_bound_interval']
        self.upper_bound_interval = params['upper_bound_interval']

        # Distraction brkaing constant
        self.distraction_constant = params['constant']

        # self.in_wave_threshold = 10

        # difference varaible for the starting vel
        self.differnce_vel_var = 0.0

        # Varaible to offset the initial positions of the cars
        self.initial_offset = 0

        # True -> perturbance on : False -> perturbance off
        self.pertubance = False

        # Initial positions and velcocties
        self.init_positions = np.zeros(self.num_vehicles)
        self.init_velocities = np.zeros(self.num_vehicles)
        # Initialize the headway (Place Holder value for now)
        self.init_headway = 0.0

        self.timeset = datetime.now().strftime("%H:%M:%S")
        self.folder = f"/Users/vincentdegaetano/Documents/VsCodeProjects/Traffic_Images/Model_1_Images/{self.timeset}_velPerturbation"

        os.makedirs(self.folder, exist_ok=True)
        self.initialize(print_yn=False)

    def update_starting_vel(self):
        self.vl0 = 12*(np.tanh(((230/22) - self.vehicle_length)/2.5-2)+np.tanh(2))/(1+np.tanh(2))  - self.differnce_vel_var


    def initialize(self, print_yn=True):

         # Initial positions and velcocties
        self.init_positions = np.zeros(self.num_vehicles)
        self.init_velocities = np.zeros(self.num_vehicles)
        # Initialize the headway (Place Holder value for now)
        self.init_headway = 0.0

        # init leader
        self.xl0 = 0.0
        self.vl0 = 30 / 3600 * 1000
        #self.vl0 = 12*(np.tanh((10.455-4)/2.5-2)+np.tanh(2))/(1+np.tanh(2)) # - 0.0001
        # Start at the steady state
        self.vl0 = 12*(np.tanh(((230/22) - self.vehicle_length)/2.5-2)+np.tanh(2))/(1+np.tanh(2))  - self.differnce_vel_var # Change to plus later
        # vl0 = 1.0
        self.init_positions[0] = self.xl0
        self.init_velocities[0] = self.vl0

        self.vel_history = np.array([])

        # init followers
        self.init_headway = float(self.params['circum'] / self.num_vehicles)
        self.v0 = 30 / 3600 * 1000
        #self.v0 = 12*(np.tanh((10.455-4)/2.5-2)+np.tanh(2))/(1+np.tanh(2))
        # Start at steady state
        self.v0 = 12*(np.tanh(((230/22) - self.vehicle_length)/2.5-2)+np.tanh(2))/(1+np.tanh(2))
        # v0 = 0.0

        self.update_starting_vel()
        if print_yn:
            print(f"{self.num_vehicles} vehicles evenly spaced {self.init_headway:.2f}m apart on ring with c={self.params['circum']}m")
            print(f"vl0 = {self.vl0}, v0 = {self.v0}")

    def dynamics(self, positions, velocities, params, time, perturbed_vehicle):
        # Set Constants
        alpha = params['alpha']
        beta = params['beta']
        vmax_desired = params['vmax_desired']
        L = params['circum']
        #vehicle_length = params['vehicle_length']

        # Initial positions and velocities for the leading vehicle
        xl = np.roll(positions, 1)
        vl = np.roll(velocities, 1)

        # Initial distance between the leading and following positions and velocities
        delta_x = (xl - positions) % L
        delta_v = vl - velocities

        # Optimal velocity
        v_optimal = vmax_desired * (np.tanh((delta_x - self.vehicle_length) / 2.5 - 2) + np.tanh(2))/(1 + np.tanh(2))

        # Return the system of differential equations
        positions_dot = velocities
        velocities_dot = alpha * (v_optimal - velocities) + beta * delta_v / delta_x ** 2

        # var = np.var(velocities_dot)

        # Hold the acceleration constant or change it based on the given parameters, on the distracted Vehicle.
        if perturbed_vehicle is not None:
            velocities_dot[perturbed_vehicle] = self.distraction_constant #velocities_dot[perturbed_vehicle] - 0.001

        return positions_dot, velocities_dot

    def rk4(self, init_positions, init_velocities, params, dt):
        # Initialize the constants
        num_steps = int(self.T / dt)
        num_vehicles = len(init_positions)

        # Initialize the positions historry and the velocity
        positions_history = np.zeros((num_steps + 1, num_vehicles))
        velocities_history = np.zeros((num_steps + 1, num_vehicles))

        perturbed_history = np.full((num_steps + 1,), None, dtype=object)

        # Set the initial position and velocity
        positions_history[0] = init_positions
        velocities_history[0] = init_velocities

        # set the current position and velocity
        positions_curr = np.array(init_positions)
        velocities_curr = np.array(init_velocities)

        # Initialize the perturbance of the vehicle when it is enabled
        perturbed_history[0] = None
        perturbed_vehicle = None
        perturbed_time_left = 0.0
        perturbation_threshold = 0.005
        has_perturbed = False
        cooldown_time_left = 0.0

        safe_gap = 5.0

        # Change the initial position by the perturbed amount
        positions_curr[0] += self.initial_offset

        # Reset the current time interval
        time_curr = 0.0

        # --- Headway calculated after this point ---

        # Loop to obtain all of the equations for the system
        for i in range(num_steps):
            # Check if the perturbance is enabled, off by default
            if self.pertubance: # and time_curr > 65.0: Starts the distractions after 65 seconds has passed
                # check headway of the perturbed car
                if perturbed_vehicle is not None:
                        leader_id = (perturbed_vehicle - 1) % num_vehicles
                        space_headway = (positions_curr[leader_id] - positions_curr[perturbed_vehicle]) % params['circum']
                        if space_headway < safe_gap:
                            
                            # Uncomment for diagnostic purposes
                            # # print(f"exiting perturbation early at {time_curr}")
                            perturbed_vehicle = None
                            perturbed_time_left = 0.0
                            cooldown_time_left = 5.0
                
                # decide perturbation
                if perturbed_time_left <= 0 and perturbed_vehicle is not None:
                    perturbed_vehicle = None
                    cooldown_time_left = 5.0

                elif perturbed_time_left <= 0 and perturbed_vehicle is None:
                    if time_curr >= 5.0 and not has_perturbed and cooldown_time_left <= 0:
                        if np.random.rand() < perturbation_threshold:
                            # perturbed_vehicle = 0 # Sets the distracted car to the leading car
                            perturbed_vehicle = np.random.randint(0, num_vehicles) # Picks a random car to apply the distraction effect to
                            perturbed_time_left = np.random.uniform(self.lower_bound_interval, self.upper_bound_interval) # Default values 2.0 - 5.0
                            # has_perturbed = True
                            # Uncomment for diagnostic purposes
                            # # print(f'perturbed at time {time_curr:.2f} with {perturbation_threshold:.3f} chance on vehicle {perturbed_vehicle} for {perturbed_time_left:.2f}s')
                
                perturbed_history[i] = perturbed_vehicle

            # K1
            k1_pos_dot, k1_vel_dot = self.dynamics(positions_curr, velocities_curr, params, time_curr, perturbed_vehicle)

            # K2
            k2_pos_dot, k2_vel_dot = self.dynamics(
                positions_curr + 0.5 * dt * k1_pos_dot,
                velocities_curr + 0.5 * dt * k1_vel_dot,
                params, time_curr + 0.5 * dt, perturbed_vehicle
            )

            # K3
            k3_pos_dot, k3_vel_dot = self.dynamics(
                positions_curr + 0.5 * dt * k2_pos_dot,
                velocities_curr + 0.5 * dt * k2_vel_dot,
                params, time_curr + 0.5 * dt, perturbed_vehicle
            )

            # K4
            k4_pos_dot, k4_vel_dot = self.dynamics(
                positions_curr + dt * k3_pos_dot,
                velocities_curr + dt * k3_vel_dot,
                params, time_curr + dt, perturbed_vehicle
            )

            # Update the current position and velocity
            positions_curr += (dt / 6) * (k1_pos_dot + 2 * k2_pos_dot + 2 * k3_pos_dot + k4_pos_dot)
            positions_curr = positions_curr % params['circum']
            velocities_curr += (dt / 6) * (k1_vel_dot + 2 * k2_vel_dot + 2 * k3_vel_dot + k4_vel_dot)
            time_curr += dt

            # print(k4_pos_dot)

            # Check if the pertubation has ended
            if perturbed_vehicle is not None:
                perturbed_time_left -= dt
            if cooldown_time_left > 0:
                cooldown_time_left -= dt

            # Update the history of the positions and velocities
            positions_history[i + 1] = positions_curr
            velocities_history[i + 1] = velocities_curr

            perturbed_history[num_steps] = perturbed_vehicle

        # Return the positions
        return np.array(positions_history), np.array(velocities_history), perturbed_history

    # for stability
    def run_simulation(self, alpha, beta, params):
        params['alpha'] = alpha
        params['beta'] = beta

        positions_history, velocities_history, perturbed_history = self.rk4(self.init_positions, self.init_velocities, params, self.dt)
        #stability_metric = np.std(velocities_history[-1, :])
        #return stability_metric

        data_var = np.var(velocities_history)
        return data_var
    
    def run_system(self, plot_yn=True):
        self.update_starting_vel()
        # Run a simulation for each car
        for i in range(1, self.num_vehicles):

            self.init_positions[i] = (self.init_positions[i - 1] - self.init_headway) % self.params['circum']
            self.init_velocities[i] = self.v0

        positions_history, velocities_history, perturbed_history = self.rk4(self.init_positions, self.init_velocities, self.params, self.dt)
        vel_std_history = np.std(velocities_history, axis=1)
        time = np.linspace(0, self.T, len(positions_history))

        self.vel_history = velocities_history

        self.wave_started.append(self.check_variance(velocities_history))

        if plot_yn:
            self.plot_figure(positions_history, velocities_history, time, vel_std_history)

    def check_variance(self, _set):
        in_wave = False
        _var = np.var(_set, axis=1)
        started = 0
        ended = 0
        for i in range(len(_set)):
            if not in_wave: #and i>50:
                if _var[i] >= self.wave_threshold:
                    print(f'Threshold reached :{_var[i]}, at time {i * self.dt}')
                    if ended == 0:
                        started = i * self.dt
                        print(f'appened {started}')
                    in_wave = True
            elif in_wave and _var[i] < self.wave_threshold:
                in_wave = False
                ended = i * self.dt

        return round(started, 3)

    def plot_figure(self, position_history, velocity_history, time, vel_std):
        if self.pertubance:
            distraction = 'distraction'
        else:
            distraction = 'NoDostraction'

        folder_path = self.folder

        print(f"Saving Figures with the extension: ..{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.png")
        # Plot positions
        plt.figure(figsize=(20, 7)) # Default is (10, 7)
        for i in range(self.num_vehicles):
            pos = position_history[:, i]
            pos_diff = np.diff(pos)
            wrap = np.where(pos_diff < -self.params['circum'] * 0.5)[0]
            pos_masked = np.array(pos, dtype=float)
            pos_masked[wrap + 1] = np.nan
            plt.plot(time, pos_masked, label=f'v{i}')
        plt.title('Positions vs Time')
        plt.xlabel('Time')
        plt.ylabel('Position m')
        plt.grid(True)
        plt.legend()
        # plt.autoscale()
        plt.tight_layout()
        name = f'{folder_path}/pos{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.png'
        plt.savefig(name)

        # Plot velocities
        plt.figure(figsize=(20, 7))
        for i in range(self.num_vehicles):
            plt.plot(time, velocity_history[:, i], label=f'v{i}')
        plt.title('Velocities vs Time')
        plt.xlabel('Time')
        plt.ylabel('Velocity m/s')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name = f'{folder_path}/vel{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.png'
        plt.savefig(name)

        # Plot relative position vs time
        plt.figure(figsize=(20, 7))
        for i in range(self.num_vehicles):
            relative_position = (position_history[:, i - 1] - position_history[:, i]) % self.params['circum']
            plt.plot(time, relative_position, label=f'v{(i-1) % self.num_vehicles} - v{i}')
        plt.title('Relative Positions vs Time')
        plt.xlabel('Time')
        plt.ylabel('Relative Position')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name = f'{folder_path}/pos_rel{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.png'
        plt.savefig(name)

        # Plot relative velocity vs time
        plt.figure(figsize=(20, 7))
        for i in range(self.num_vehicles):
            relative_velocity = (velocity_history[:, i - 1] - velocity_history[:, i])
            plt.plot(time, relative_velocity, label=f'v{(i-1) % self.num_vehicles} - v{i}')
        plt.title('Relative Velocities vs Time')
        plt.xlabel('Time')
        plt.ylabel('Relative Velocity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name = f'{folder_path}/vel_rel{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.png'
        plt.savefig(name)

        # Statistics
        steps_per_second = int(1.0 / self.dt)
        time_seconds = time[::steps_per_second]
        std_vel_seconds = vel_std[::steps_per_second]
        csv_filename = f'{folder_path}/statistics_{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.csv'
        table_data = np.column_stack((time_seconds, std_vel_seconds))
        np.savetxt(
            csv_filename, 
            table_data, 
            delimiter=',', 
            header='Time (s),Std Dev Velocity (m/s)', 
            comments='',
            fmt='%.1f'
        )
        plt.figure(figsize=(20, 7))
        plt.plot(time, vel_std, label="Velocity Statistics")
        plt.title('Std vel vs time')
        plt.xlabel('Time s')
        plt.ylabel('Std')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_path}/statistics_{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.png')

        # Plot the wave_started time vs the velocity_perturbations
        # Plot wave started vs vel_perturbation 
        # if len(self.wave_started) >= self.num_vehicles:
        #     plt.figure(figsize=(20, 7))
        #     for i in range(self.num_vehicles):
        #         relative_velocity = (velocity_history[self.wave_started[self.wave_placeholder]:, i - 1] - velocity_history[self.wave_started[self.wave_placeholder]:, i])
        #         plt.plot(time[self.wave_started[self.wave_placeholder] :], relative_velocity) # label=f'v{(i-1) % self.num_vehicles} - v{i}'
        #     plt.title('Relative Velocities vs Time, After wave Started')
        #     plt.xlabel('Time')
        #     plt.ylabel('Relative Velocity')
        #     plt.grid(True)
        #     #plt.legend()
        #     plt.tight_layout()
        #     name = f'/Users/vincentdegaetano/Documents/VsCodeProjects/Traffic_Images/Model_1_Images/Wave_Started_rel_vel{self.differnce_vel_var:0.2f}_{distraction}_{self.timeset}.png'
        #     plt.savefig(name)


# Collect the start time
start_time = time.perf_counter()


params = {
    'alpha': 1.0, #0.5, # 1.0
    'beta': 10.45 **2, ##20.0,##10.45 ** 2, # 20.0,
    'vmax_desired': 12.0, # 40 / 3600 * 1000,
    'circum': 230.0,
    'vehicle_length': 4.0,
    'perturbed_time': 2.0,
    'lower_bound_interval': 2.0,
    'upper_bound_interval': 5.0,
    'constant': 0.0
}


manager = BandoFtl(params, 1000)
manager.pertubance = False # Distraction effect

# values = [-0.001, -0.005, -0.01, -0.1, -0.5, -1.0, -1.5, -2, -2.1, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 1.5, 2, 2.1]
## values = [-2.1, -2.0, -1.5, -1.0, -0.5, -0.1, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 1.5, 2, 2.1] Don't need to include the zero case
values = [-2.1, -2.0, -1.5, -1.0, -0.5, -0.1, -0.01, -0.005, -0.001, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 1.5, 2, 2.1]

step_count = 100

values = np.linspace(-2.1, 2.1, step_count)

# values = [0]

# Set the time to save the graphs as:
timeset = datetime.now()
timeset = timeset.strftime("%H:%M:%S")

values = [0.0]

print(values)
#values = [0.01]
step = 0
go = True
if go:
    for i in values:
        # print(i)
        manager.differnce_vel_var = i
        manager.initialize()
        manager.run_system(plot_yn=True)
        
        # For printing
        print(f"The wave started at: {manager.wave_started[step]}s; Threshold: {manager.wave_threshold}; Velocity Perturbation: {i}")

        if manager.wave_started[step] == 0:
            manager.wave_started.remove(0)

        plt.close('all')
        step += 1

end_time = time.perf_counter()

execution_time = end_time - start_time

print(f'\nSimulation completed in {execution_time:.02f}s, ({execution_time / 60:.02f} mins)')
print(f"The initial velocity perturbations of the leading vehicle's velocity were the range [{values[0]}, {values[-1]}], with a step count of {step_count}.")
on = ''
if manager.pertubance:
    on = 'on'
else:
    on = 'off'
print(f"The waves have started at the following time intervals with vehicle distractions turned {on}")
print(manager.wave_started)
# The values that are graphed from running the simulation:
# 144.68, 147.31, 163.16, 186.29, 227.46, 326.68, 471.1, 514.6800000000001, 615.9300000000001, 0, 615.92, 514.63, 470.99, 325.64, 222.25, 175.91, 147.57, 126.68, 123.04

# Plot the Velocuty Perturbation vs. Wave Started time
if len(values) >= 2:
    plot = True
else:
    plot = False

if plot:
    if manager.pertubance:
        distractionf = f'Distraction'
    else:
        distractionf = f'NoDistraction'

    plt.figure(figsize=(20, 7))
    y = manager.wave_started
    plt.plot(values, y)

    plt.title('Wave Started vs Velocity Perturbation')
    plt.xlabel('Velocity Perturbation (m/s)')
    plt.ylabel('Time Wave Started (s)')
    plt.grid(True)
    plt.tight_layout()
    namef = f'{manager.folder}/WaveStartedvsPerturbation_{values[-1]}_{distractionf}_{timeset}.png'
    plt.savefig(namef)

    print(f'Saved Figure as {namef}')

today = datetime.today()

# Create the info.txt file
with open(f"{manager.folder}/info.txt", "w") as file:
    file.write(f"Simulation completed in {execution_time:.02f}s, ({execution_time / 60:.02f} mins). \nThe initial velocity perturbations of the leading vehicle's velocity were the range [{values[0]}, {values[-1]}], with a step count of {step_count}. \nThe waves have started at the following time intervals with vehicle distractions turned {on}. \n{manager.wave_started}")
    file.write(f"\nThe parameters used are as follows: \n {params}\n")
    file.write(f"The simulation was completed on {today}")

plt.close('all')