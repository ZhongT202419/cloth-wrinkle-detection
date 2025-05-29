import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# -------------------------------
# Mass-Spring Simulation Classes
# -------------------------------

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

# Simulate one time step with improved stability
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

    # Update positions using Verlet integration with limits
    MAX_VELOCITY = 1000.0
    MAX_POSITION = 1e6

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

                # Limit velocity and position
                velocity_norm = np.linalg.norm(point.velocity)
                if velocity_norm > MAX_VELOCITY:
                    point.velocity = (point.velocity / velocity_norm) * MAX_VELOCITY
                position_norm = np.linalg.norm(point.position)
                if position_norm > MAX_POSITION:
                    point.position = (point.position / position_norm) * MAX_POSITION

                # Additional check for finite values
                if not np.all(np.isfinite(point.position)):
                    print("Warning: Non-finite position detected.")
                    point.position = np.nan_to_num(point.position)
                if not np.all(np.isfinite(point.velocity)):
                    print("Warning: Non-finite velocity detected.")
                    point.velocity = np.nan_to_num(point.velocity)

# -------------------------------------
# Wrinkle Detection and Correction
# -------------------------------------

def compute_curvature(mass_points):
    curvature_map = []
    for i in range(len(mass_points)):
        row_curvature = []
        for j in range(len(mass_points[0])):
            point = mass_points[i][j]
            neighbors = []

            # Get neighboring points
            if i > 0:
                neighbors.append(mass_points[i - 1][j].position)
            if i < len(mass_points) - 1:
                neighbors.append(mass_points[i + 1][j].position)
            if j > 0:
                neighbors.append(mass_points[i][j - 1].position)
            if j < len(mass_points[0]) - 1:
                neighbors.append(mass_points[i][j + 1].position)

            # Compute curvature
            if neighbors:
                neighbor_positions = np.array(neighbors)
                average_neighbor = np.mean(neighbor_positions, axis=0)
                curvature = np.linalg.norm(point.position - average_neighbor)
            else:
                curvature = 0.0
            row_curvature.append(curvature)
        curvature_map.append(row_curvature)
    return np.array(curvature_map)

def apply_geometric_smoothing(mass_points, curvature_map, curvature_threshold, smoothing_factor):
    for i in range(len(mass_points)):
        for j in range(len(mass_points[0])):
            point = mass_points[i][j]
            if point.fixed:
                continue

            curvature = curvature_map[i][j]
            # Threshold to identify wrinkles
            if curvature > curvature_threshold:
                neighbors = []

                # Get neighboring points
                if i > 0:
                    neighbors.append(mass_points[i - 1][j].position)
                if i < len(mass_points) - 1:
                    neighbors.append(mass_points[i + 1][j].position)
                if j > 0:
                    neighbors.append(mass_points[i][j - 1].position)
                if j < len(mass_points[0]) - 1:
                    neighbors.append(mass_points[i][j + 1].position)

                # Compute corrective force
                neighbor_positions = np.array(neighbors)
                average_neighbor = np.mean(neighbor_positions, axis=0)
                corrective_force = smoothing_factor * (average_neighbor - point.position)
                point.force += corrective_force

# Placeholder functions for image processing (normal mapping, displacement mapping)
def apply_normal_mapping():
    pass

def apply_displacement_mapping():
    pass

# -------------------------------------
# Reinforcement Learning Environment
# -------------------------------------

class ClothEnv(gym.Env):
    def __init__(self, active_components):
        super(ClothEnv, self).__init__()
        self.active_components = active_components
        # Define action and observation space
        # Actions: Adjustments to fabric tension (e.g., stiffness) and damping
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observations: Flattened positions and velocities of mass points
        self.grid_width = 10   # Adjust as needed
        self.grid_height = 10  # Adjust as needed
        observation_dim = self.grid_width * self.grid_height * 3 * 2  # positions and velocities
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

        self.reset()

    def reset(self):
        # Reset the simulation to initial state
        grid_spacing = 1.0  # Distance between mass points
        mass = 1.0  # Mass of each point
        stiffness = 5000.0  # Initial spring stiffness
        damping = 5.0  # Initial damping coefficient

        # Define fixed points (e.g., the top edge of the cloth)
        fixed_points = [(0, j) for j in range(self.grid_width)]
        # Initialize mass points
        self.mass_points = initialize_cloth(self.grid_width, self.grid_height, grid_spacing, mass, fixed_points)
        # Create springs
        self.stiffness = stiffness
        self.damping = damping
        self.springs = create_springs(self.mass_points, self.stiffness, self.damping)

        self.time_step = 0
        self.max_time_steps = 100
        self.done = False

        # Return initial observation
        observation = self._get_observation()
        return observation

    def step(self, action):
        # Adjust for DummyVecEnv wrapping
        if isinstance(action, np.ndarray) and action.ndim == 2:
            action = action[0]  # Extract action from array

        # Apply action (adjust fabric tension and damping)
        self._apply_action(action)

        # Simulate one time step
        simulate_step(self.mass_points, self.springs, dt=0.001)

        # Optionally, apply geometric smoothing
        if 'geometric_smoothing' in self.active_components:
            curvature_map = compute_curvature(self.mass_points)
            apply_geometric_smoothing(self.mass_points, curvature_map, curvature_threshold=0.1, smoothing_factor=0.5)

        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            self.done = True

        # Compute reward
        reward = self._compute_reward()

        # Get next observation
        observation = self._get_observation()

        return observation, reward, self.done, {}

    def _get_observation(self):
        positions = np.array([point.position for row in self.mass_points for point in row]).flatten()
        velocities = np.array([point.velocity for row in self.mass_points for point in row]).flatten()
        observation = np.concatenate([positions, velocities])
        return observation

    def _apply_action(self, action):
        # Ensure action is a numpy array
        action = np.array(action)
        # Adjust fabric stiffness and damping based on action
        stiffness_adjustment = action[0] * 1000  # Scale adjustment
        damping_adjustment = action[1] * 1       # Scale adjustment

        self.stiffness += stiffness_adjustment
        self.damping += damping_adjustment

        # Clamp stiffness and damping to positive values
        self.stiffness = max(0, self.stiffness)
        self.damping = max(0, self.damping)

        # Update spring properties
        for spring in self.springs:
            spring.stiffness = self.stiffness
            spring.damping = self.damping

    def _compute_reward(self):
        # Compute curvature map
        curvature_map = compute_curvature(self.mass_points)
        # Reward is negative average curvature (we want to minimize curvature)
        average_curvature = np.mean(curvature_map)
        reward = -average_curvature
        return reward

# -------------------------------------
# Ablation Studies
# -------------------------------------

def perform_ablation_studies():
    components = ['geometric_smoothing', 'reinforcement_learning']
    baseline_performance = evaluate_system(components)
    print(f"Baseline performance with all components: {baseline_performance}")

    for component in components:
        # Remove the component
        ablated_components = [c for c in components if c != component]
        performance = evaluate_system(ablated_components)
        print(f"Performance without {component}: {performance}")

def evaluate_system(active_components):
    # Initialize environment with active components
    env = ClothEnv(active_components)

    # Load or train RL agent if reinforcement_learning is active
    if 'reinforcement_learning' in active_components:
        env = DummyVecEnv([lambda: env])
        model = PPO('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=10000)
        use_model = True
    else:
        model = None
        use_model = False

    # Run simulation
    total_reward = 0
    obs = env.reset()
    max_time_steps = env.get_attr('max_time_steps')[0] if use_model else env.max_time_steps

    for _ in range(max_time_steps):
        if use_model:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            done = dones[0]
        else:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward

        if done:
            break

    average_reward = total_reward / max_time_steps
    return average_reward

# -------------------------------------
# Cross-Validation Techniques
# -------------------------------------

def cross_validation_runs(num_runs=3):
    performances = []
    for run in range(num_runs):
        print(f"Cross-validation run {run + 1}/{num_runs}")
        performance = evaluate_system(['geometric_smoothing', 'reinforcement_learning'])
        performances.append(performance)
    average_performance = np.mean(performances)
    print(f"Average performance over {num_runs} runs: {average_performance}")

# -------------------------------------
# Integration into Simulation
# -------------------------------------

def simulation_with_wrinkle_correction():
    active_components = ['geometric_smoothing', 'reinforcement_learning']
    env = ClothEnv(active_components)
    env = DummyVecEnv([lambda: env])

    # Train RL agent
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save('cloth_rl_agent')

    # Run simulation with trained agent
    obs = env.reset()
    max_time_steps = env.get_attr('max_time_steps')[0]

    for _ in range(max_time_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, infos = env.step(action)
        if dones[0]:
            break

    print("Simulation with wrinkle correction completed.")

# -------------------------------------
# Main Execution
# -------------------------------------

def main():
    # Perform ablation studies
    perform_ablation_studies()

    # Cross-validation runs
    cross_validation_runs(num_runs=3)

    # Simulation with wrinkle correction
    simulation_with_wrinkle_correction()

if __name__ == "__main__":
    main()
