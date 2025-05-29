import os
import numpy as np
import matplotlib.pyplot as plt

# Define the MassPoint class
class MassPoint:
    def __init__(self, position, mass, fixed=False):
        self.position = np.array(position, dtype=float)
        self.previous_position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.force = np.zeros(3)
        self.mass = mass
        self.fixed = fixed

# Define the Spring class
class Spring:
    def __init__(self, point_a, point_b, stiffness, damping):
        self.point_a = point_a
        self.point_b = point_b
        self.rest_length = np.linalg.norm(point_b.position - point_a.position)
        self.stiffness = stiffness
        self.damping = damping

# Initialize the cloth grid
def initialize_cloth(width, height, grid_spacing, mass, fixed_points):
    mass_points = []
    for i in range(height):
        row = []
        for j in range(width):
            position = [j * grid_spacing, 0, i * grid_spacing]
            fixed = (i, j) in fixed_points
            point = MassPoint(position, mass, fixed)
            row.append(point)
        mass_points.append(row)
    return mass_points

# Create springs between mass points
def create_springs(mass_points, stiffness, damping):
    springs = []
    height = len(mass_points)
    width = len(mass_points[0])

    # Structural Springs
    for i in range(height):
        for j in range(width):
            if j < width - 1:
                spring = Spring(mass_points[i][j], mass_points[i][j+1], stiffness, damping)
                springs.append(spring)
            if i < height - 1:
                spring = Spring(mass_points[i][j], mass_points[i+1][j], stiffness, damping)
                springs.append(spring)

    # Shear Springs
    for i in range(height - 1):
        for j in range(width - 1):
            spring = Spring(mass_points[i][j], mass_points[i+1][j+1], stiffness, damping)
            springs.append(spring)
            spring = Spring(mass_points[i+1][j], mass_points[i][j+1], stiffness, damping)
            springs.append(spring)

    # Bend Springs
    for i in range(height):
        for j in range(width):
            if j < width - 2:
                spring = Spring(mass_points[i][j], mass_points[i][j+2], stiffness * 0.5, damping)
                springs.append(spring)
            if i < height - 2:
                spring = Spring(mass_points[i][j], mass_points[i+2][j], stiffness * 0.5, damping)
                springs.append(spring)
    return springs

# Compute spring forces with epsilon check
def compute_spring_force(spring):
    pos_a = spring.point_a.position
    pos_b = spring.point_b.position
    delta = pos_b - pos_a
    distance = np.linalg.norm(delta)

    # Small epsilon to prevent division by zero
    epsilon = 1e-8
    if distance < epsilon:
        direction = np.zeros(3)
        force_magnitude = 0.0
        damping_force = np.zeros(3)
    else:
        direction = delta / distance
        force_magnitude = -spring.stiffness * (distance - spring.rest_length)
        relative_velocity = spring.point_b.velocity - spring.point_a.velocity
        damping_force = -spring.damping * np.dot(relative_velocity, direction) * direction

    force = force_magnitude * direction + damping_force
    return force

# Simulate one time step
def simulate_step(mass_points, springs, dt):
    # Reset forces
    for row in mass_points:
        for point in row:
            point.force = np.zeros(3)
            if not point.fixed:
                # Gravity
                point.force += np.array([0, -9.81 * point.mass, 0])

    # Compute spring forces
    for spring in springs:
        force = compute_spring_force(spring)
        if not spring.point_a.fixed:
            spring.point_a.force += force
        if not spring.point_b.fixed:
            spring.point_b.force -= force

    # Update positions using Verlet integration
    for row in mass_points:
        for point in row:
            if not point.fixed:
                acceleration = point.force / point.mass
                temp_position = np.copy(point.position)
                point.position = (2 * point.position) - point.previous_position + acceleration * dt * dt
                point.previous_position = temp_position
                # Simple damping
                point.velocity = (point.position - point.previous_position) / dt
                point.velocity *= 0.99  # Damping factor

                # Check for NaN or Inf in position and velocity
                if not np.all(np.isfinite(point.position)):
                    print("Warning: Non-finite position detected.")
                    point.position = np.nan_to_num(point.position)
                if not np.all(np.isfinite(point.velocity)):
                    print("Warning: Non-finite velocity detected.")
                    point.velocity = np.nan_to_num(point.velocity)

# Visualization function with custom colours
def save_cloth_plot(mass_points, springs, step, save_dir):
    x_coords = [point.position[0] for row in mass_points for point in row]
    y_coords = [point.position[1] for row in mass_points for point in row]
    z_coords = [point.position[2] for row in mass_points for point in row]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Cloth Simulation at Step {step}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize colours
    mass_point_color = 'red'     # Change to desired colour for mass points
    spring_color = 'green'       # Change to desired colour for springs

    # Plot the mass points
    ax.scatter(x_coords, y_coords, z_coords, color=mass_point_color, s=5)

    # Plot the springs
    for spring in springs:
        x = [spring.point_a.position[0], spring.point_b.position[0]]
        y = [spring.point_a.position[1], spring.point_b.position[1]]
        z = [spring.point_a.position[2], spring.point_b.position[2]]
        ax.plot(x, y, z, color=spring_color, linewidth=0.5)

    ax.set_xlim(0, grid_width * grid_spacing)
    ax.set_ylim(-grid_height * grid_spacing, 0)
    ax.set_zlim(0, grid_height * grid_spacing)

    # Save the plot
    filename = os.path.join(save_dir, f'step_{step:04d}.png')
    plt.savefig(filename)
    plt.close(fig)

# Main simulation function
def run_simulation():
    global grid_width, grid_height, grid_spacing
    grid_width = 20   # Number of mass points along width
    grid_height = 20  # Number of mass points along height
    grid_spacing = 1.0  # Distance between mass points
    mass = 1.0  # Mass of each point
    stiffness = 5000.0  # Spring stiffness
    damping = 1.0  # Spring damping

    # Define fixed points (e.g., the top edge of the cloth)
    fixed_points = [(0, j) for j in range(grid_width)]
    # Initialize mass points
    mass_points = initialize_cloth(grid_width, grid_height, grid_spacing, mass, fixed_points)
    # Create springs
    springs = create_springs(mass_points, stiffness, damping)
    # Simulation parameters
    time_steps = 200
    dt = 0.01  # Time step duration

    # Directory to save plots
    save_dir = r'E:\Clothing Wrinkles\Images\spring'  # Update with your desired path
    os.makedirs(save_dir, exist_ok=True)

    for step in range(time_steps):
        simulate_step(mass_points, springs, dt)
        if step % 10 == 0:
            print(f"Simulation step {step}")
            save_cloth_plot(mass_points, springs, step, save_dir)
    print("Simulation completed.")

if __name__ == "__main__":
    run_simulation()
