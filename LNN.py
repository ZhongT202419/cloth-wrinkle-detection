import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# Data Validation Function
# -------------------------------------

def validate_data(X_data, Y_data):
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Check for NaN or Inf
    if not np.all(np.isfinite(X_data)):
        raise ValueError("X_data contains non-finite values.")
    if not np.all(np.isfinite(Y_data)):
        raise ValueError("Y_data contains non-finite values.")

    # Check for excessively large values
    if np.any(np.abs(X_data) > 1e6):
        raise ValueError("X_data contains values that are too large.")
    if np.any(np.abs(Y_data) > 1e6):
        raise ValueError("Y_data contains values that are too large.")

    return X_data, Y_data

# -------------------------------------
# Data Preparation
# -------------------------------------

def prepare_data(X_data, Y_data, sequence_length):
    # Convert lists to NumPy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Validate data
    X_data, Y_data = validate_data(X_data, Y_data)

    # Normalize features using StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_data = scaler_X.fit_transform(X_data)
    Y_data = scaler_Y.fit_transform(Y_data)

    # Create sequences
    X_sequences = []
    Y_sequences = []
    for i in range(len(X_data) - sequence_length):
        X_sequences.append(X_data[i:i+sequence_length])
        Y_sequences.append(Y_data[i:i+sequence_length])

    X_sequences = np.array(X_sequences)
    Y_sequences = np.array(Y_sequences)

    return X_sequences, Y_sequences, scaler_X, scaler_Y

# -------------------------------------
# LSTM Model Definition
# -------------------------------------

def build_lstm_model(sequence_length, num_features, output_size):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(sequence_length, num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(units=output_size)))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# -------------------------------------
# Model Training with Cross-Validation
# -------------------------------------

def train_model_with_cv(model_func, X_data, Y_data, n_splits=5):
    kf = KFold(n_splits=n_splits)
    fold = 0
    histories = []

    for train_index, val_index in kf.split(X_data):
        fold += 1
        print(f"Training on fold {fold}/{n_splits}")

        X_train_cv, X_val_cv = X_data[train_index], X_data[val_index]
        Y_train_cv, Y_val_cv = Y_data[train_index], Y_data[val_index]

        model = model_func()
        history = model.fit(
            X_train_cv, Y_train_cv,
            validation_data=(X_val_cv, Y_val_cv),
            epochs=50,
            batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
            verbose=1
        )
        histories.append((model, history))

    return histories

# -------------------------------------
# Model Evaluation
# -------------------------------------

def evaluate_model(model, X_test, Y_test, scaler_Y):
    Y_test_pred = model.predict(X_test)

    # Reshape for metric calculations
    output_size = Y_test.shape[2]
    Y_test_true = Y_test.reshape(-1, output_size)
    Y_test_pred = Y_test_pred.reshape(-1, output_size)

    # Inverse transform the outputs
    Y_test_true = scaler_Y.inverse_transform(Y_test_true)
    Y_test_pred = scaler_Y.inverse_transform(Y_test_pred)

    # Check for non-finite values in predictions
    if not np.all(np.isfinite(Y_test_pred)):
        raise ValueError("Predictions contain non-finite values.")

    mse = mean_squared_error(Y_test_true, Y_test_pred)
    mae = mean_absolute_error(Y_test_true, Y_test_pred)
    r2 = r2_score(Y_test_true, Y_test_pred)

    print(f'Test MSE: {mse}')
    print(f'Test MAE: {mae}')
    print(f'Test R^2: {r2}')

    return Y_test_true, Y_test_pred

# -------------------------------------
# Simulation and Data Collection
# -------------------------------------

def run_simulation_and_collect_data(simulations=5):
    X_data = []
    Y_data = []

    for sim in range(simulations):
        print(f"Running simulation {sim + 1}/{simulations}")
        global grid_width, grid_height, grid_spacing
        grid_width = 10   # Adjust as needed
        grid_height = 10  # Adjust as needed
        grid_spacing = 1.0  # Distance between mass points
        mass = 1.0  # Mass of each point
        stiffness = 5000.0  # Spring stiffness
        damping = 5.0  # Increased damping

        # Define fixed points (e.g., the top edge of the cloth)
        fixed_points = [(0, j) for j in range(grid_width)]
        # Initialize mass points
        mass_points = initialize_cloth(grid_width, grid_height, grid_spacing, mass, fixed_points)
        # Create springs
        springs = create_springs(mass_points, stiffness, damping)
        # Simulation parameters
        time_steps = 100  # Adjust as needed
        dt = 0.001  # Reduced time step duration

        # Material properties vector (constant in this case)
        material_properties_vector = np.array([stiffness, damping, mass])

        # External forces vector (gravity in this case)
        external_forces_vector = np.array([0, -9.81 * mass, 0])

        for step in range(time_steps):
            simulate_step(mass_points, springs, dt)

            # Collect input features
            # Flatten the positions and velocities of all mass points
            current_positions = np.array([point.position for row in mass_points for point in row]).flatten()
            current_velocities = np.array([point.velocity for row in mass_points for point in row]).flatten()
            input_features = np.concatenate([
                material_properties_vector,
                external_forces_vector,
                current_positions,
                current_velocities
            ])
            X_data.append(input_features)

            # Collect output labels (next positions)
            simulate_step(mass_points, springs, dt)  # Advance to next step
            next_positions = np.array([point.position for row in mass_points for point in row]).flatten()
            Y_data.append(next_positions)

            # Revert back to current state (if necessary)
            # simulate_step_back(mass_points, springs, dt)

            if step % 20 == 0:
                print(f"Simulation {sim + 1}, step {step}")

    return X_data, Y_data

# -------------------------------------
# Integration into Simulation
# -------------------------------------

def run_simulation_with_lstm(model, scaler_X, scaler_Y, sequence_length):
    global grid_width, grid_height, grid_spacing
    grid_width = 10   # Match the training grid size
    grid_height = 10  # Match the training grid size
    grid_spacing = 1.0  # Distance between mass points
    mass = 1.0  # Mass of each point
    stiffness = 5000.0  # Spring stiffness
    damping = 5.0  # Damping coefficient

    # Define fixed points (e.g., the top edge of the cloth)
    fixed_points = [(0, j) for j in range(grid_width)]
    # Initialize mass points
    mass_points = initialize_cloth(grid_width, grid_height, grid_spacing, mass, fixed_points)

    # Simulation parameters
    time_steps = 100  # Adjust as needed
    dt = 0.001  # Time step duration

    # Material properties vector (constant in this case)
    material_properties_vector = np.array([stiffness, damping, mass])

    # External forces vector (gravity in this case)
    external_forces_vector = np.array([0, -9.81 * mass, 0])

    # Initialize sequence buffer
    current_sequence = []

    for step in range(time_steps):
        # Collect input features
        current_positions = np.array([point.position for row in mass_points for point in row]).flatten()
        current_velocities = np.array([point.velocity for row in mass_points for point in row]).flatten()
        input_features = np.concatenate([
            material_properties_vector,
            external_forces_vector,
            current_positions,
            current_velocities
        ])
        input_features_scaled = scaler_X.transform(input_features.reshape(1, -1)).flatten()
        current_sequence.append(input_features_scaled)

        if len(current_sequence) > sequence_length:
            current_sequence.pop(0)

        if len(current_sequence) == sequence_length:
            # Prepare input for prediction
            X_input = np.array([current_sequence])
            Y_pred = model.predict(X_input)
            next_positions_scaled = Y_pred[0, -1]
            next_positions = scaler_Y.inverse_transform(next_positions_scaled.reshape(1, -1)).flatten()

            # Update mass point positions with predicted positions
            positions = next_positions.reshape(-1, 3)
            idx = 0
            for i in range(grid_height):
                for j in range(grid_width):
                    if not mass_points[i][j].fixed:
                        mass_points[i][j].position = positions[idx]
                    idx += 1

        # Optional: Save plots or perform other operations

        if step % 10 == 0:
            print(f"LSTM Simulation at step {step}")

    # After simulation, you can visualize or analyze the results

# -------------------------------------
# Visualization Functions
# -------------------------------------

def plot_predicted_vs_actual(Y_true, Y_pred, sample_index=0):
    plt.figure(figsize=(10, 6))
    plt.plot(Y_true[sample_index], label='Actual')
    plt.plot(Y_pred[sample_index], label='Predicted')
    plt.title('Predicted vs Actual Positions')
    plt.xlabel('Position Index')
    plt.ylabel('Position Value')
    plt.legend()
    plt.show()

def plot_error_distribution(Y_true, Y_pred):
    errors = Y_true - Y_pred
    plt.hist(errors.flatten(), bins=50)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.show()

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# -------------------------------------
# Main Execution
# -------------------------------------

def main():
    # Step 1: Run simulation and collect data
    X_data, Y_data = run_simulation_and_collect_data(simulations=5)

    # Step 2: Prepare data
    sequence_length = 10  # Adjust as needed
    X_sequences, Y_sequences, scaler_X, scaler_Y = prepare_data(X_data, Y_data, sequence_length)

    # Step 3: Build model function
    num_features = X_sequences.shape[2]
    output_size = Y_sequences.shape[2]
    def model_func():
        return build_lstm_model(sequence_length, num_features, output_size)

    # Step 4: Train model with cross-validation
    histories = train_model_with_cv(model_func, X_sequences, Y_sequences, n_splits=5)

    # Choose the model with the lowest validation loss
    best_model_index = np.argmin([min(history.history['val_loss']) for _, history in histories])
    best_model = histories[best_model_index][0]
    best_history = histories[best_model_index][1]

    # Step 5: Evaluate model
    # Split data into training and test sets
    _, X_test, _, Y_test = train_test_split(X_sequences, Y_sequences, test_size=0.2, random_state=42)
    Y_test_true, Y_test_pred = evaluate_model(best_model, X_test, Y_test, scaler_Y)

    # Step 6: Visualize predictions
    plot_predicted_vs_actual(Y_test_true, Y_test_pred, sample_index=0)
    plot_error_distribution(Y_test_true, Y_test_pred)
    plot_training_history(best_history)

    # Step 7: Save the trained model
    best_model.save('lstm_wrinkle_model.h5')
    print("Model saved to 'lstm_wrinkle_model.h5'.")

    # Step 8: Run simulation with LSTM predictions
    # To load the model:
    # from tensorflow.keras.models import load_model
    # best_model = load_model('lstm_wrinkle_model.h5')
    run_simulation_with_lstm(best_model, scaler_X, scaler_Y, sequence_length)

if __name__ == "__main__":
    main()
