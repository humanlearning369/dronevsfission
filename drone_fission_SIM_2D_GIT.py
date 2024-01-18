"""
Fission Energy Drone Simulation

Authors: Frank Garcia & Naomi Arroyo
Collaborator: n/a
Date: 01/02/2024
License: MIT LICENSE

Description:
This script simulates drone monitoring and management in a fission energy environment.
It includes visualization of energy spread and drone interaction with the energy field.

Usage:
Run this script to simulate the drone's reactions to an evolving energy field and to visualize the process.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pycharge as pc
from numpy import linspace, meshgrid
from scipy.constants import e
import meep as mp
import random

SAFETY_THRESHOLD = 1000
RESOURCE_MANAGEMENT_AMOUNT = 100 # runs for 20 min, increasing this value will increase time to run

# grid size, fission point
grid_size = 100
fission_point = (50, 50)

# param
total_time = 30
peak_time = 10  # reached energy peak
spread_speed = 2  # energy spread (controlled)

# calc energy at a grid point
def calculate_energy(x, y, time):
    # calc distance from the fission point
    distance = np.sqrt((x - fission_point[0])**2 + (y - fission_point[1])**2)
    # calc energy spread
    spread_radius = spread_speed * time
    spread_radius = max(spread_radius, 1e-6) # added 1/12
    # if the point is within the spread radius
    within_radius = distance <= spread_radius
    # calc energy
    energy_peak = np.exp(-((time - peak_time)**2) / (2 * (peak_time / 2)**2))
    # adjust energy based on distance (if within the spread radius)
    energy = energy_peak * np.exp(-distance**2 / (2 * (spread_radius / 2)**2)) if within_radius else 0
    return energy

# energy spread visualization
def plot_energy_spread_animation(drones, total_time, grid_size, energy_distributions):
    fig, ax = plt.subplots()
    grid = np.zeros((grid_size, grid_size))
    cax = ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(cax)

    drone_plots = [ax.plot([], [], 'bo')[0] for _ in drones]

    def update(frame):
        # energy grid ref
        current_energy_grid = energy_distributions[frame]
        cax.set_data(current_energy_grid)

        # drone positions to avoid collission
        all_drones_positions = {tuple(drone.position) for drone in drones}

        # drone positions and sim
        for drone, drone_plot in zip(drones, drone_plots):
            drone.react_to_energy_grid(current_energy_grid, all_drones_positions)
            drone.check_safety()
            drone.manage_resources()
            drone.display_info()
            drone_plot.set_data([drone.position[1]], [drone.position[0]])

        ax.set_title(f"Time: {frame}")
        return [cax] + drone_plots

    # animation
    ani = FuncAnimation(fig, update, frames=total_time + 1, interval=200, blit=True, repeat=False)
    total_animation_duration = total_time * 200  # 200 is the interval in milliseconds
    timer = fig.canvas.new_timer(interval=total_animation_duration + 1000)  # +1000 to give some extra time
    timer.add_callback(plt.close, fig)
    timer.start()    
    plt.show()

# class and functions
class Drone:
   def __init__(self, identifier, field_strength, water_cooling_capacity, position=(0, 0), battery_level=100, shielding_material='hydrogen-rich polymer', size=(1, 1, 1)):
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
       # if drone field strength below threshold then drone is safe else move pos
       if self.field_strength < SAFETY_THRESHOLD:
           print(f"Drone {self.identifier} is safe.")
       else:
           print(f"Drone {self.identifier} is unsafe.")

   def manage_resources(self):       
       # is need to reduce the field strength by measure?
       self.field_strength -= RESOURCE_MANAGEMENT_AMOUNT

   def calculate_cooling_effect(self, temperature):        
        # is the cooling effect proportional to the temperature?
        cooling_effect = self.water_cooling_capacity * temperature
        return cooling_effect

   def find_highest_energy_point(self, energy_grid, search_radius):
       x, y = int(self.position[0]), int(self.position[1])
       max_energy = 0
       target = None

       # search the energy grid radius
       for dx in range(-search_radius, search_radius + 1):
           for dy in range(-search_radius, search_radius + 1):
               if 0 <= x + dx < grid_size and 0 <= y + dy < grid_size:
                   if energy_grid[x + dx, y + dy] > max_energy:
                       max_energy = energy_grid[x + dx, y + dy]
                       target = (dx, dy)

       return target
       
   # is the target position occupied by another drone? change pos
   def can_move_to(self, position, all_drones_positions):       
       return tuple(position) not in all_drones_positions

   def react_to_energy_grid(self, energy_grid, all_drones_positions, search_radius=5):
       target = self.find_highest_energy_point(energy_grid, search_radius)

       if target and target != (0, 0):
           move_direction = np.sign(target)
           new_position = self.position + move_direction
           if self.can_move_to(new_position, all_drones_positions):
               self.move(move_direction)


# materials can be changed in segments where applicable. hrp is most effective for scenario
materials = {
    'lead': {'atomic_number': 82, 'density': 11.34, 'light_hydrogen_content': 0},
    'aluminum': {'atomic_number': 13, 'density': 2.7, 'light_hydrogen_content': 0},
    'polyethylene': {'atomic_number': 1, 'density': 0.96, 'light_hydrogen_content': 0},
    'concrete': {'atomic_number': 1, 'density': 2.4, 'light_hydrogen_content': 0},
    'hydrogen-rich polymer': {'atomic_number': 1, 'density': 0.9, 'light_hydrogen_content': 1}
}

# drone mgmt and sim
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

# energy dist per time step
energy_distributions = []
for time_step in range(total_time):
    energy_grid = np.zeros((grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            energy_grid[x, y] = calculate_energy(x, y, time_step)
    energy_distributions.append(energy_grid)


# drone sim
def run_drone_simulation():
    target_energy = 10000
    containment_threshold = 15000

    drones = [Drone(i, 100 + i * 5, 50 + i * 2,
                    position=(random.uniform(0, grid_size - 1), random.uniform(0, grid_size - 1)),
                    shielding_material=materials['hydrogen-rich polymer'], size=(1, 1, 1)) for i in range(1, 151)]

    for time_step in range(total_time):
        current_energy_grid = energy_distributions[time_step]

        # param all drone positions to avoid collission
        all_drones_positions = {tuple(drone.position) for drone in drones}

        for drone in drones:
            drone.react_to_energy_grid(current_energy_grid, all_drones_positions)
            drone.check_safety()  # drone safety status
            drone.manage_resources()  # drone resources
            drone.display_info()  # updated drone info            
            
            if check_containment_success(drones, target_energy, containment_threshold):
                print(f"Containment successful at time step {time_step}")
                break  # exit

            # update drone positions after movement
            all_drones_positions = {tuple(drone.position) for drone in drones}

    # pycharge sim
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

    # dim
    sx, sy, sz = 50, 50, 50  # Cell size dimensions (in arbitrary units)
    resolution = 10  # Sample resolution

    # meep sim (a)
    drone_material = mp.Medium(epsilon=12)

    geometry = []
    for drone in drones:
        drone_size = mp.Vector3(1, 1, 1)
        drone_position = mp.Vector3(*drone.position, 0)
        geometry.append(mp.Block(center=drone_position, size=drone_size, material=drone_material))

    # cont (a)
    cell_size = mp.Vector3(sx, sy, sz)
    sim = mp.Simulation(cell_size=cell_size, geometry=geometry, resolution=resolution)

    # cont (a)
    sim.run(until=200)

    # fission sim (event)
    distribute_fission_energy(drones, (50, 50), 10000)
    simulate_containment(drones, 10000)
    drones[0].move((10, 10))
    drones[0].adjust_field_strength(200)
    simulate_environmental_impact(drones, 10000, 300)

    for drone in drones:
        drone.check_safety()
        drone.display_info()    
        drone.manage_resources()
        drone.display_info()

    # if containment success then print drone info
    containment_threshold = 15000
    success = check_containment_success(drones, 10000, containment_threshold)
    if success:
        print("\nContainment successfully completed.")
        for drone in drones:
            print(f"Drone ID: {drone.identifier}, Position: {drone.position}, Field Strength: {drone.field_strength}, Water Cooling: {drone.water_cooling_capacity}")

# main
if __name__ == '__main__':    
    drones = [Drone(i, 100 + i * 5, 50 + i * 2,
                    position=(random.uniform(0, grid_size - 1), random.uniform(0, grid_size - 1)),
                    shielding_material=materials['hydrogen-rich polymer'], size=(1, 1, 1)) for i in range(1, 151)]

    plot_energy_spread_animation(drones, total_time, grid_size, energy_distributions)    
    run_drone_simulation()
