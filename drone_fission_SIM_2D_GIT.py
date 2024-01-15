import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pycharge as pc
from numpy import linspace, meshgrid
from scipy.constants import e
import meep as mp
import random

SAFETY_THRESHOLD = 1000 # Set this to whatever value makes sense for your application
RESOURCE_MANAGEMENT_AMOUNT = 100 # Set this to whatever value makes sense for your application

# Grid size and fission point
grid_size = 100
fission_point = (50, 50)

# Time parameters
total_time = 30
peak_time = 10  # Time at which the energy peak is reached
spread_speed = 2  # Controls how fast the energy spreads

# Function to calculate energy at a grid point
def calculate_energy(x, y, time):
    # Calculate distance from the fission point
    distance = np.sqrt((x - fission_point[0])**2 + (y - fission_point[1])**2)
    # Calculate the radius of the energy spread
    spread_radius = spread_speed * time
    spread_radius = max(spread_radius, 1e-6) # added 1/12
    # Determine if the point is within the spread radius
    within_radius = distance <= spread_radius
    # Calculate energy using a Gaussian distribution
    energy_peak = np.exp(-((time - peak_time)**2) / (2 * (peak_time / 2)**2))
    # Adjust energy based on distance (if within the spread radius)
    energy = energy_peak * np.exp(-distance**2 / (2 * (spread_radius / 2)**2)) if within_radius else 0
    return energy

# Plotting and animation function for energy spread visualization
def plot_energy_spread_animation(drones, total_time, grid_size, energy_distributions):
    fig, ax = plt.subplots()
    grid = np.zeros((grid_size, grid_size))
    cax = ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(cax)

    drone_plots = [ax.plot([], [], 'bo')[0] for _ in drones]  # Initial empty plots for drones

# def plot_energy_spread_animation():
#    # Initialize the grid
#    grid = np.zeros((grid_size, grid_size))
#    fig, ax = plt.subplots()
#    cax = ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
#    fig.colorbar(cax)

    def update(frame):
        # Update energy grid
        current_energy_grid = energy_distributions[frame]
        cax.set_data(current_energy_grid)

        # Collect all drone positions for collision avoidance
        all_drones_positions = {tuple(drone.position) for drone in drones}

        # Update drone positions and process simulation logic
        for drone, drone_plot in zip(drones, drone_plots):
            drone.react_to_energy_grid(current_energy_grid, all_drones_positions)
            drone.check_safety()
            drone.manage_resources()
            drone.display_info()
            drone_plot.set_data([drone.position[1]], [drone.position[0]])

        ax.set_title(f"Time: {frame}")
        return [cax] + drone_plots

   # Update function for animation
# def update(frame):
#    nonlocal grid # Declare grid as a nonlocal variable
#    for x in range(grid_size):
#        for y in range(grid_size):
#            grid[x, y] = calculate_energy(x, y, frame)
#    cax.set_data(grid)
#    ax.set_title(f"Time: {frame}")
#    return cax,
#
    # Create animation
    ani = FuncAnimation(fig, update, frames=total_time + 1, interval=200, blit=True, repeat=False)

# code below added 1/12
    total_animation_duration = total_time * 200  # 200 is the interval in milliseconds
    timer = fig.canvas.new_timer(interval=total_animation_duration + 1000)  # +1000 to give some extra time
    timer.add_callback(plt.close, fig)
    timer.start()
    # Show the animation
    plt.show()

# Definitions for the Drone class and related functions

# Drone class and its methods
class Drone:
   def __init__(self, identifier, field_strength, water_cooling_capacity, position=(0, 0), battery_level=100, shielding_material='lead', size=(1, 1, 1)):
       self.identifier = identifier
       self.field_strength = field_strength
       self.water_cooling_capacity = water_cooling_capacity
       self.size = size
       self.position = np.array(position)
       self.battery_level = battery_level
       self.shielding_material = shielding_material

   def display_info(self):
       print(f"Drone ID: {self.identifier}, Position: {self.position}, Field Strength: {self.field_strength}, Water Cooling: {self.water_cooling_capacity}")

   def move(self, delta):
       self.position += np.array(delta)

   def adjust_field_strength(self, new_strength):
       self.field_strength = new_strength

   def check_safety(self):
       # Implement your safety check logic here
       # For example, if the drone is safe if its field strength is below a certain threshold:
       if self.field_strength < SAFETY_THRESHOLD:
           print(f"Drone {self.identifier} is safe.")
       else:
           print(f"Drone {self.identifier} is unsafe.")

   def manage_resources(self):
       # Implement your resource management logic here
       # For example, if managing resources means reducing the field strength by a certain amount:
       self.field_strength -= RESOURCE_MANAGEMENT_AMOUNT

   def calculate_cooling_effect(self, temperature):
        # Implement your cooling calculation logic here
        # For example, if the cooling effect is proportional to the temperature:
        cooling_effect = self.water_cooling_capacity * temperature
        return cooling_effect

   def find_highest_energy_point(self, energy_grid, search_radius):
       x, y = int(self.position[0]), int(self.position[1])
       max_energy = 0
       target = None

       # Search within the specified radius
       for dx in range(-search_radius, search_radius + 1):
           for dy in range(-search_radius, search_radius + 1):
               if 0 <= x + dx < grid_size and 0 <= y + dy < grid_size:
                   if energy_grid[x + dx, y + dy] > max_energy:
                       max_energy = energy_grid[x + dx, y + dy]
                       target = (dx, dy)

       return target

   def can_move_to(self, position, all_drones_positions):
       # Check if the target position is occupied by another drone
       return tuple(position) not in all_drones_positions

   def react_to_energy_grid(self, energy_grid, all_drones_positions, search_radius=5):
       target = self.find_highest_energy_point(energy_grid, search_radius)

       if target and target != (0, 0):
           move_direction = np.sign(target)
           new_position = self.position + move_direction
           if self.can_move_to(new_position, all_drones_positions):
               self.move(move_direction)


# Global variables and other functions
materials = {
    'lead': {'atomic_number': 82, 'density': 11.34, 'light_hydrogen_content': 0},
    'aluminum': {'atomic_number': 13, 'density': 2.7, 'light_hydrogen_content': 0},
    'polyethylene': {'atomic_number': 1, 'density': 0.96, 'light_hydrogen_content': 0},
    'concrete': {'atomic_number': 1, 'density': 2.4, 'light_hydrogen_content': 0},
    'hydrogen-rich polymer': {'atomic_number': 1, 'density': 0.9, 'light_hydrogen_content': 1}
}

# Additional functions for drone management and simulations
def calculate_dynamic_electric_field(drones, target_energy):
    field_effect = {}
    for drone in drones:
        distance = (target_energy / drone.field_strength) ** 0.5
        field_effect[drone.identifier] = 1 / distance ** 2
    return field_effect

def simulate_containment(drones, target_energy):
    field_impacts = calculate_dynamic_electric_field(drones, target_energy)
    total_cooling = sum(drone.water_cooling_capacity for drone in drones)
    print(f"Total cooling capacity: {total_cooling}")
    for drone_id, impact in field_impacts.items():
        print(f"Drone {drone_id} field impact on target: {impact}")

def feedback_loop(drones, target_energy, actual_containment):
    for drone in drones:
        if actual_containment < target_energy:
            drone.adjust_field_strength(drone.field_strength + 10)
        elif actual_containment > target_energy:
            drone.adjust_field_strength(drone.field_strength - 10)

def simulate_environmental_impact(drones, target_energy, ambient_temperature):
    total_field_strength = sum(drone.field_strength for drone in drones)
    total_cooling_effect = sum(drone.calculate_cooling_effect(ambient_temperature) for drone in drones)
    environmental_impact = (target_energy / total_field_strength) - total_cooling_effect
    print(f"Estimated Environmental Impact: {environmental_impact}")

def distribute_fission_energy(drones, fission_point, total_fission_energy):
    for drone in drones:
        drone.field_strength += total_fission_energy / len(drones)

def check_containment_success(drones, target_energy, containment_threshold):
   total_field_strength = sum(drone.field_strength for drone in drones)
   if total_field_strength >= containment_threshold:
       print("Containment successful!")
       return True
   return False

# Code to generate energy distributions for each time step
energy_distributions = []
for time_step in range(total_time):
    energy_grid = np.zeros((grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            energy_grid[x, y] = calculate_energy(x, y, time_step)
    energy_distributions.append(energy_grid)


# Main function to execute the drone simulation
def run_drone_simulation():
    target_energy = 10000  # Define an appropriate value for your simulation
    containment_threshold = 15000  # Define an appropriate value for your simulation

    drones = [Drone(i, 100 + i * 5, 50 + i * 2,
                    position=(random.uniform(0, grid_size - 1), random.uniform(0, grid_size - 1)),
                    shielding_material=materials['lead'], size=(1, 1, 1)) for i in range(1, 151)]


    for time_step in range(total_time):
        current_energy_grid = energy_distributions[time_step]

        # Collect all drone positions for collision avoidance
        all_drones_positions = {tuple(drone.position) for drone in drones}

        for drone in drones:
            drone.react_to_energy_grid(current_energy_grid, all_drones_positions)
            drone.check_safety()  # Check safety status of each drone
            drone.manage_resources()  # Manage resources for each drone
            drone.display_info()  # Display updated drone information

            # Add any additional simulation-wide checks or updates here
            # Example: Check for containment success
            if check_containment_success(drones, target_energy, containment_threshold):
                print(f"Containment successful at time step {time_step}")
                break  # Exit the simulation loop if successful

            # Update the set of all drone positions after they have moved
            all_drones_positions = {tuple(drone.position) for drone in drones}

    # PyCharge simulation setup
    sources = []
    for drone in drones:
        origin = drone.position
        charge = drone.field_strength
        origin = np.append(origin, 0)
        sources.append(pc.StationaryCharge(origin, charge))

    simulation = pc.Simulation(sources)
    coord = linspace(-50e-9, 50e-9, 1001)
    x, y, z = meshgrid(coord, coord, 0, indexing='ij')
    Ex, Ey, Ez = simulation.calculate_E(0, x, y, z)

    # Sample dimensions for the Meep simulation
    sx, sy, sz = 50, 50, 50  # Cell size dimensions (in arbitrary units)
    resolution = 10  # Sample resolution

    # Meep simulation setup
    drone_material = mp.Medium(epsilon=12)

    geometry = []
    for drone in drones:
        drone_size = mp.Vector3(1, 1, 1)
        drone_position = mp.Vector3(*drone.position, 0)
        geometry.append(mp.Block(center=drone_position, size=drone_size, material=drone_material))

    # Create the Meep simulation
    cell_size = mp.Vector3(sx, sy, sz)
    sim = mp.Simulation(cell_size=cell_size, geometry=geometry, resolution=resolution)

    # Run the Meep simulation
    sim.run(until=200)

    # Fission event simulation and additional functions
    distribute_fission_energy(drones, (50, 50), 10000)
    simulate_containment(drones, 10000)
    drones[0].move((10, 10))
    drones[0].adjust_field_strength(200)
    simulate_environmental_impact(drones, 10000, 300)

    for drone in drones:
        drone.check_safety()
        drone.display_info()

    for drone in drones:
        drone.manage_resources()
        drone.display_info()

    # Check containment success and print drone info if successful
    containment_threshold = 15000
    success = check_containment_success(drones, 10000, containment_threshold)
    if success:
        print("\nContainment successfully completed.")
        for drone in drones:
            print(f"Drone ID: {drone.identifier}, Position: {drone.position}, Field Strength: {drone.field_strength}, Water Cooling: {drone.water_cooling_capacity}")

# Main execution
if __name__ == '__main__':
    # Run the plot animation from Code 1
    # plot_energy_spread_animation()
    drones = [Drone(i, 100 + i * 5, 50 + i * 2,
                    position=(random.uniform(0, grid_size - 1), random.uniform(0, grid_size - 1)),
                    shielding_material=materials['lead'], size=(1, 1, 1)) for i in range(1, 151)]

    plot_energy_spread_animation(drones, total_time, grid_size, energy_distributions)

    # Run the drone simulation from Code 2
    run_drone_simulation()