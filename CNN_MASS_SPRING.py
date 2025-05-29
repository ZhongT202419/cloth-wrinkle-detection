import numpy as np
import matplotlib.pyplot as plt
import os

# Simulation parameters
GRID_WIDTH = 30  # Increased grid width
GRID_HEIGHT = 30  # Increased grid height
MASS = 1.0  # Mass of each point
SPRING_STIFFNESS = 20.0  # Significantly reduced stiffness of the springs
DAMPING = 0.9  # Reduced damping for more oscillation
TIME_STEP = 0.01  # Time step for the simulation
GRAVITY = np.array([0, -50.0])  # Stronger gravity to exaggerate wrinkle effects
WIND_FORCE = np.array([15.0, 0.0])  # External wind force applied horizontally

# Create a grid of mass points
class MassPoint:
    def __init__(self, position, pinned=False):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0, 0], dtype=float)
        self.force = np.array([0, 0], dtype=float)
        self.pinned = pinned  # Whether the mass point is fixed or free

    def update_position(self, time_step):
        if not self.pinned:
            acceleration = self.force / MASS
            self.velocity = DAMPING * (self.velocity + acceleration * time_step)
            self.position += self.velocity * time_step

# Create a spring class to connect mass points
class Spring:
    def __init__(self, p1, p2, rest_length):
        self.p1 = p1
        self.p2 = p2
        self.rest_length = rest_length

    def compute_force(self):
        # Hooke's law for spring force
        displacement = self.p2.position - self.p1.position
        distance = np.linalg.norm(displacement)
        direction = displacement / distance if distance != 0 else np.zeros_like(displacement)
        force_magnitude = SPRING_STIFFNESS * (distance - self.rest_length)
        force = direction * force_magnitude

        # Apply equal and opposite forces to the connected points
        self.p1.force += force
        self.p2.force -= force

# Initialize grid of mass points and springs
def initialize_fabric(grid_width, grid_height, rest_length):
    mass_points = []
    for x in range(grid_width):
        row = []
        for y in range(grid_height):
            pinned = (x == 0 and y == 0) or (x == grid_width - 1 and y == 0)  # Pin corners of the fabric
            row.append(MassPoint([x * rest_length, y * rest_length], pinned=pinned))
        mass_points.append(row)

    springs = []
    # Connect adjacent points with springs
    for x in range(grid_width):
        for y in range(grid_height):
            if x < grid_width - 1:  # Horizontal springs
                springs.append(Spring(mass_points[x][y], mass_points[x + 1][y], rest_length))
            if y < grid_height - 1:  # Vertical springs
                springs.append(Spring(mass_points[x][y], mass_points[x][y + 1], rest_length))
            if x < grid_width - 1 and y < grid_height - 1:  # Diagonal springs
                springs.append(Spring(mass_points[x][y], mass_points[x + 1][y + 1], np.sqrt(2) * rest_length))
                springs.append(Spring(mass_points[x + 1][y], mass_points[x][y + 1], np.sqrt(2) * rest_length))

    return mass_points, springs

# Function to apply external forces such as gravity and wind
def apply_external_forces(mass_points, gravity, wind_force, step):
    for row in mass_points:
        for point in row:
            if not point.pinned:
                point.force += MASS * gravity
                if step % 200 == 0:  # Apply wind force periodically
                    point.force += wind_force

# Function to update the position and velocity of mass points
def update_positions(mass_points, springs, time_step, step):
    # Reset forces to zero
    for row in mass_points:
        for point in row:
            point.force = np.array([0, 0], dtype=float)

    # Compute forces from springs
    for spring in springs:
        spring.compute_force()

    # Apply external forces (gravity and wind)
    apply_external_forces(mass_points, GRAVITY, WIND_FORCE, step)

    # Update positions and velocities using Euler integration
    for row in mass_points:
        for point in row:
            point.update_position(time_step)

# Function to display and save the grid of mass points as .png files
def display_fabric(mass_points, step_number):
    x_coords = np.array([[p.position[0] for p in row] for row in mass_points])
    y_coords = np.array([[p.position[1] for p in row] for row in mass_points])
    plt.plot(x_coords, y_coords, 'bo-', alpha=0.7)

    # Save plot to file with a unique name
    output_filename = f"mass_spring_step_{step_number:04d}.png"
    plt.savefig(output_filename)

    plt.clf()  # Clear the current figure for the next plot

# Main simulation loop
def simulate_fabric(num_steps, time_step):
    rest_length = 1.5  # Increased rest length to make the fabric more flexible
    mass_points, springs = initialize_fabric(GRID_WIDTH, GRID_HEIGHT, rest_length)

    # Create directory to save plots
    if not os.path.exists("mass_spring_plots"):
        os.makedirs("mass_spring_plots")
    
    # Change directory to save plots
    os.chdir("mass_spring_plots")

    for step in range(num_steps):
        update_positions(mass_points, springs, time_step, step)

        # Save the fabric plot every 50 steps for finer analysis
        if step % 50 == 0:
            display_fabric(mass_points, step)

    # Change back to the original directory
    os.chdir("..")

# Run the simulation for 2000 steps
simulate_fabric(num_steps=2000, time_step=TIME_STEP)
