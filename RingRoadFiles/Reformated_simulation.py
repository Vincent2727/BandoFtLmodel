import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

class BandoFtl:

    def __init__(self, parameters, time_horizon):
        """Class Manager for the BandoFtl model"""
        self.params = parameters
        self.T = time_horizon
        self.dt = 0.01
        self.num_vehicles = 22

        # True -> perturbance on : False -> perturbance off
        self.pertubance = False

        # Initial positions and velcocties
        self.init_positions = np.zeros(self.num_vehicles)
        self.init_velocities = np.zeros(self.num_vehicles)
        # Initialize the headway (Place Holder value for now)
        self.init_headway = 0.0

        self.initialize()

    def initialize(self):
        # init leader
        self.xl0 = 0.0
        self.vl0 = 30 / 3600 * 1000
        self.vl0 = 12*(np.tanh((10.455-4)/2.5-2)+np.tanh(2))/(1+np.tanh(2)) - 0.0001
        # vl0 = 1.0
        self.init_positions[0] = self.xl0
        self.init_velocities[0] = self.vl0


        # init followers
        self.init_headway = float(self.params['circum'] / self.num_vehicles)
        self.v0 = 30 / 3600 * 1000
        self.v0 = 12*(np.tanh((10.455-4)/2.5-2)+np.tanh(2))/(1+np.tanh(2))
        # v0 = 0.0

        print(f"{self.num_vehicles} vehicles evenly spaced {self.init_headway:.2f}m apart on ring with c={self.params['circum']}m")
        print(f"vl0 = {self.vl0}, v0 = {self.v0}")

    def dynamics(self, positions, velocities, params, time):
        # Set Constants
        alpha = params['alpha']
        beta = params['beta']
        vmax_desired = params['vmax_desired']
        L = params['circum']
        vehicle_length = params['vehicle_length']

        # Initial positions and velocities for the leading vehicle
        xl = np.roll(positions, 1)
        vl = np.roll(velocities, 1)

        # Initial distance between the leading and following positions and velocities
        delta_x = (xl - positions) % L
        delta_v = vl - velocities

        # Optimal velocity
        v_optimal = vmax_desired * (np.tanh((delta_x - vehicle_length) / 2.5 - 2) + np.tanh(2))/(1 + np.tanh(2))

        # Return the system of differential equations
        positions_dot = velocities
        velocities_dot = alpha * (v_optimal - velocities) + beta * delta_v / delta_x ** 2
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


        # Reset the current time interval
        time_curr = 0.0

        # Loop to obtain all of the equations for the system
        for i in range(num_steps):
            # Check if the perturbance is enabled, off by default
            if self.pertubance:
                # check headway of the perturbed car
                if perturbed_vehicle is not None:
                        leader_id = (perturbed_vehicle - 1) % num_vehicles
                        space_headway = (positions_curr[leader_id] - positions_curr[perturbed_vehicle]) % params['circum']
                        if space_headway < safe_gap:
                            print(f"exiting perturbation early at {time_curr}")
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
                            perturbed_vehicle = np.random.randint(0, num_vehicles)
                            perturbed_time_left = np.random.uniform(2.0, 5.0)
                            # has_perturbed = True
                            print(f'perturbed at time {time_curr:.2f} with {perturbation_threshold:.3f} chance on vehicle {perturbed_vehicle} for {perturbed_time_left:.2f}s')
                
                perturbed_history[i] = perturbed_vehicle

            # K1
            k1_pos_dot, k1_vel_dot = self.dynamics(positions_curr, velocities_curr, params, time_curr)

            # K2
            k2_pos_dot, k2_vel_dot = self.dynamics(
                positions_curr + 0.5 * dt * k1_pos_dot,
                velocities_curr + 0.5 * dt * k1_vel_dot,
                params, time_curr + 0.5 * dt
            )

            # K3
            k3_pos_dot, k3_vel_dot = self.dynamics(
                positions_curr + 0.5 * dt * k2_pos_dot,
                velocities_curr + 0.5 * dt * k2_vel_dot,
                params, time_curr + 0.5 * dt
            )

            # K4
            k4_pos_dot, k4_vel_dot = self.dynamics(
                positions_curr + dt * k3_pos_dot,
                velocities_curr + dt * k3_vel_dot,
                params, time_curr + dt
            )

            # Update the current position and velocity
            positions_curr += (dt / 6) * (k1_pos_dot + 2 * k2_pos_dot + 2 * k3_pos_dot + k4_pos_dot)
            positions_curr = positions_curr % params['circum']
            velocities_curr += (dt / 6) * (k1_vel_dot + 2 * k2_vel_dot + 2 * k3_vel_dot + k4_vel_dot)
            time_curr += dt

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
    
    def run_system(self, v0):
        # Run a simulation for each car
        for i in range(1, self.num_vehicles):

            self.init_positions[i] = (self.init_positions[i - 1] - self.init_headway) % self.params['circum']
            self.init_velocities[i] = v0

        positions_history, velocities_history, perturbed_history = self.rk4(self.init_positions, self.init_velocities, self.params, self.dt)
        vel_std_history = np.std(velocities_history, axis=1)
        time = np.linspace(0, self.T, len(positions_history))

        self.plot_figure(positions_history, velocities_history, time, vel_std_history)


    def plot_figure(self, position_history, velocity_history, time, vel_std):
        # Plot positions
        plt.figure(figsize=(10, 7))
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
        plt.tight_layout()
        plt.savefig('pos.png')

        # Plot velocities
        plt.figure(figsize=(10, 7))
        for i in range(self.num_vehicles):
            plt.plot(time, velocity_history[:, i], label=f'v{i}')
        plt.title('Velocities vs Time')
        plt.xlabel('Time')
        plt.ylabel('Velocity m/s')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('vel.png')

        # Plot relative position vs time
        plt.figure(figsize=(10, 7))
        for i in range(self.num_vehicles):
            relative_position = (position_history[:, i - 1] - position_history[:, i]) % self.params['circum']
            plt.plot(time, relative_position, label=f'v{(i-1) % self.num_vehicles} - v{i}')
        plt.title('Relative Positions vs Time')
        plt.xlabel('Time')
        plt.ylabel('Relative Position')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('pos_rel.png')

        # Plot relative velocity vs time
        plt.figure(figsize=(10, 7))
        for i in range(self.num_vehicles):
            relative_velocity = (velocity_history[:, i - 1] - velocity_history[:, i])
            plt.plot(time, relative_velocity, label=f'v{(i-1) % self.num_vehicles} - v{i}')
        plt.title('Relative Velocities vs Time')
        plt.xlabel('Time')
        plt.ylabel('Relative Velocity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('vel_rel.png')

        # Statistics
        steps_per_second = int(1.0 / self.dt)
        time_seconds = time[::steps_per_second]
        std_vel_seconds = vel_std[::steps_per_second]
        csv_filename = 'statistics.csv'
        table_data = np.column_stack((time_seconds, std_vel_seconds))
        np.savetxt(
            csv_filename, 
            table_data, 
            delimiter=',', 
            header='Time (s),Std Dev Velocity (m/s)', 
            comments='',
            fmt='%.1f'
        )
        plt.figure(figsize=(10, 7))
        plt.plot(time, vel_std, label="Velocity Statistics")
        plt.title('Std vel vs time')
        plt.xlabel('Time s')
        plt.ylabel('Std')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('statistics.png')

params = {
    'alpha': 0.5,
    'beta': 20.0,
    'vmax_desired': 12.0, # 40 / 3600 * 1000,
    'circum': 230.0,
    'vehicle_length': 4.0,
    'perturbed_time': 2.0,
}


manager = BandoFtl(params, 500)

manager.run_system(manager.v0)

print(manager.run_simulation(0.5, 20.0, params))
