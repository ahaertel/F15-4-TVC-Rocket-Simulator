
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons

# --------------------------------------------------
# Motor specifications
# --------------------------------------------------

total_impulse = 49.61       # N*s
burn_time = 3.45            # s
ejection_delay = 4.0        # s
motor_mass = 0.1018         # kg
propellant_mass = 0.060     # kg

# Rocket specifications
rocket_mass = 0.253         # kg, excluding motor
initial_mass = rocket_mass + motor_mass
rocket_diameter = 0.050     # m
drag_coefficient = 0.75

# Physics
gravity = 9.81              # m/s^2
air_density = 1.225         # kg/m^3
scale_height = 8500.0       # m
dt = 0.01                   # s
max_time = 120.0            # s

# TVC settings
max_angle = 10.0            # degrees
transition_time = 0.5       # s, illustrative

area = np.pi * (rocket_diameter / 2)**2
average_thrust = total_impulse / burn_time

# (Time in seconds, commanded angle in degrees)
profiles = {
    "Fixed 0°": [(0.0, 0.0)],
    "Fixed 2°": [(0.0, 2.0)],
    "Fixed 5°": [(0.0, 5.0)],
    "Fixed 10°": [(0.0, 10.0)],
    "Early adjustment": [(0.0, 5.0), (1.0, 0.0)],
    "Late adjustment": [(0.0, 0.0), (2.0, 5.0)],
    "Direction reversal": [(0.0, 5.0), (1.5, -5.0)]
}


# --------------------------------------------------
# Smooth TVC angle functions
# --------------------------------------------------

def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3*u**2 - 2*u**3


def get_command_angle(t, profile):
    angle = 0.0

    for change_time, target_angle in profile:
        if t >= change_time:
            angle = target_angle
        else:
            break

    return np.clip(angle, -max_angle, max_angle)


def make_smooth_angle_function(profile):
    """
    Precalculate smooth transitions.

    Each new transition starts from the actual angle
    reached when its command begins.
    """

    transitions = []
    previous_angle = 0.0

    for change_time, target_angle in profile:
        target_angle = np.clip(
            target_angle, -max_angle, max_angle
        )

        if transitions:
            previous_transition = transitions[-1]

            start_time = previous_transition[0]
            start_angle = previous_transition[1]
            previous_target = previous_transition[2]

            elapsed = change_time - start_time
            u = elapsed / transition_time

            previous_angle = start_angle + (
                previous_target - start_angle
            ) * smoothstep(u)

        transitions.append((
            change_time,
            previous_angle,
            target_angle
        ))

    def angle_at_time(t):
        angle = 0.0

        for change_time, start_angle, target_angle in transitions:
            if t < change_time:
                break

            u = (t - change_time) / transition_time

            angle = start_angle + (
                target_angle - start_angle
            ) * smoothstep(u)

        return angle

    return angle_at_time


# --------------------------------------------------
# Force calculations
# --------------------------------------------------

def calculate_forces(t, position, velocity, angle):

    fuel_used = propellant_mass * min(
        t / burn_time, 1.0
    )

    mass = initial_mass - fuel_used

    thrust = (
        average_thrust
        if t < burn_time
        else 0.0
    )

    theta = np.radians(angle)

    # Thrust vector (N)
    thrust_force = thrust * np.array([
        np.sin(theta),
        np.cos(theta)
    ])

    # Air density (kg/m^3)
    density = air_density * np.exp(
        -max(position[1], 0.0) / scale_height
    )

    # Drag vector (N)
    speed = np.linalg.norm(velocity)

    drag_force = (
        -0.5 * density * drag_coefficient
        * area * speed * velocity
    )

    # Gravity vector (N)
    gravity_force = np.array([
        0.0,
        -mass * gravity
    ])

    # Net force (N)
    net_force = (
        thrust_force
        + drag_force
        + gravity_force
    )

    # Acceleration (m/s^2)
    acceleration = net_force / mass

    return {
        "mass": mass,
        "thrust": thrust_force,
        "gravity": gravity_force,
        "drag": drag_force,
        "net": net_force,
        "acceleration": acceleration
    }


# --------------------------------------------------
# Rocket simulation
# --------------------------------------------------

def simulate(profile):

    smooth_angle = make_smooth_angle_function(profile)

    position = np.array([0.0, 0.0])  # m
    velocity = np.array([0.0, 0.0])  # m/s

    times = [0.0]
    path = [position.copy()]
    velocities = [velocity.copy()]

    actual_angles = [smooth_angle(0.0)]
    command_angles = [get_command_angle(0.0, profile)]

    launched = False

    for step in range(int(max_time / dt)):

        t = step * dt

        # Smooth actual angle (degrees)
        angle = smooth_angle(t)

        # Forces at the current position
        forces = calculate_forces(
            t, position, velocity, angle
        )

        acceleration = forces["acceleration"]

        previous_position = position.copy()

        # Update velocity and position
        velocity += acceleration * dt
        position += velocity * dt

        if position[1] > 0:
            launched = True

        landed = launched and position[1] <= 0

        if landed:
            denominator = (
                previous_position[1] - position[1]
            )

            fraction = (
                previous_position[1] / denominator
            )

            position = previous_position + fraction * (
                position - previous_position
            )

            position[1] = 0.0

        elif not launched and position[1] < 0:
            position[1] = 0.0
            velocity[1] = 0.0

        current_time = (
            t + fraction * dt
            if landed
            else t + dt
        )

        times.append(current_time)
        path.append(position.copy())
        velocities.append(velocity.copy())

        actual_angles.append(
            smooth_angle(current_time)
        )

        command_angles.append(
            get_command_angle(current_time, profile)
        )

        if landed:
            break

    times = np.array(times)
    path = np.array(path)
    velocities = np.array(velocities)
    actual_angles = np.array(actual_angles)
    command_angles = np.array(command_angles)

    thrust_forces = []
    gravity_forces = []
    drag_forces = []
    net_forces = []
    accelerations = []
    masses = []

    for t, pos, vel, angle in zip(
        times, path, velocities, actual_angles
    ):

        forces = calculate_forces(
            t, pos, vel, angle
        )

        thrust_forces.append(forces["thrust"])
        gravity_forces.append(forces["gravity"])
        drag_forces.append(forces["drag"])
        net_forces.append(forces["net"])
        accelerations.append(forces["acceleration"])
        masses.append(forces["mass"])

    return {
        "times": times,
        "path": path,
        "velocities": velocities,
        "angles": actual_angles,
        "commands": command_angles,
        "thrust": np.array(thrust_forces),
        "gravity": np.array(gravity_forces),
        "drag": np.array(drag_forces),
        "net": np.array(net_forces),
        "accelerations": np.array(accelerations),
        "masses": np.array(masses)
    }


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

results = {
    name: simulate(profile)
    for name, profile in profiles.items()
}

print("\nF15-4 TVC Experiment Results")
print("-" * 65)

for name, result in results.items():

    path = result["path"]
    times = result["times"]

    print(
        f"{name:20s} | "
        f"Landing X: {path[-1, 0]:7.2f} m | "
        f"Apogee: {np.max(path[:, 1]):7.2f} m | "
        f"Flight: {times[-1]:6.2f} s"
    )

# --------------------------------------------------
# Verify vector calculations
# --------------------------------------------------

def verify_vectors(results):

    print("\nVector Verification")
    print("-" * 65)

    for name, result in results.items():

        thrust = result["thrust"]
        gravity = result["gravity"]
        drag = result["drag"]
        net = result["net"]

        acceleration = result["accelerations"]
        masses = result["masses"]

        times = result["times"]
        angles = result["angles"]

        # Vector addition
        expected_net = thrust + gravity + drag

        net_error = np.max(
            np.linalg.norm(
                net - expected_net,
                axis=1
            )
        )

        # Thrust magnitude
        expected_magnitude = np.where(
            times < burn_time,
            average_thrust,
            0.0
        )

        thrust_magnitude = np.linalg.norm(
            thrust, axis=1
        )

        thrust_error = np.max(
            np.abs(
                thrust_magnitude - expected_magnitude
            )
        )

        # Thrust components
        expected_fx = (
            expected_magnitude
            * np.sin(np.radians(angles))
        )

        expected_fy = (
            expected_magnitude
            * np.cos(np.radians(angles))
        )

        expected_components = np.column_stack(
            (expected_fx, expected_fy)
        )

        component_error = np.max(
            np.abs(thrust - expected_components)
        )

        # Newton's second law
        newton_error = np.max(
            np.linalg.norm(
                net - masses[:, None] * acceleration,
                axis=1
            )
        )

        tolerance = 1e-8

        passed = all([
            net_error < tolerance,
            thrust_error < tolerance,
            component_error < tolerance,
            newton_error < tolerance
        ])

        print(
            f"{name:20s} | "
            f"Net: {net_error:.2e} | "
            f"Thrust: {thrust_error:.2e} | "
            f"Components: {component_error:.2e} | "
            f"Newton: {newton_error:.2e} | "
            f"{'PASS' if passed else 'FAIL'}"
        )


verify_vectors(results)


# --------------------------------------------------
# Animated visualization
# --------------------------------------------------

fig = plt.figure(figsize=(15, 8))

ax_path = fig.add_axes([0.26, 0.16, 0.38, 0.70])
ax_force = fig.add_axes([0.70, 0.43, 0.26, 0.40])
ax_info = fig.add_axes([0.70, 0.10, 0.26, 0.27])
ax_select = fig.add_axes([0.02, 0.30, 0.20, 0.48])

selector = RadioButtons(
    ax_select,
    list(profiles.keys())
)

ax_info.axis("off")

# Flight path
path_line, = ax_path.plot(
    [], [], "b-", lw=2, label="Flight Path"
)

rocket_dot, = ax_path.plot(
    [], [], "bo", markersize=8
)

landing_dot, = ax_path.plot(
    [], [], "go", markersize=8, label="Landing"
)

burnout_dot, = ax_path.plot(
    [], [],
    marker="x",
    color="red",
    markersize=10,
    markeredgewidth=2,
    linestyle="None",
    label="Motor Burnout"
)

ax_path.set_xlabel("Horizontal Position (m)")
ax_path.set_ylabel("Altitude (m)")
ax_path.grid()
ax_path.legend(loc="upper right")

# Force-vector diagram
ax_force.set_title("Force Vectors (N)")
ax_force.set_xlabel("Horizontal Force (N)")
ax_force.set_ylabel("Vertical Force (N)")
ax_force.set_aspect("equal", adjustable="box")
ax_force.grid()

force_colors = {
    "Thrust": "green",
    "Gravity": "red",
    "Drag": "orange",
    "Net": "purple"
}

force_arrows = {}

for name, color in force_colors.items():

    force_arrows[name] = ax_force.quiver(
        [0], [0], [0], [0],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        label=name
    )

ax_force.legend(
    loc="upper left",
    fontsize=8
)

# Thrust components
fx_line, = ax_force.plot(
    [], [], "b--", lw=1.5
)

fy_line, = ax_force.plot(
    [], [], "m--", lw=1.5
)

# Motor status
motor_status = fig.text(
    0.70, 0.92, "",
    fontsize=12,
    fontweight="bold"
)

# Live numerical information
info_text = ax_info.text(
    0.0, 1.0, "",
    va="top",
    family="monospace",
    fontsize=9,
    transform=ax_info.transAxes
)

current_name = list(profiles.keys())[0]
current_result = results[current_name]

frame_index = 0
animation_step = 5


def set_limits():

    path = current_result["path"]

    x_min = min(0.0, np.min(path[:, 0]))
    x_max = max(0.0, np.max(path[:, 0]))
    y_max = np.max(path[:, 1])

    padding = max(
        5.0,
        (x_max - x_min) * 0.1
    )

    ax_path.set_xlim(
        x_min - padding,
        x_max + padding
    )

    ax_path.set_ylim(
        0,
        max(10.0, y_max * 1.1)
    )

    all_forces = np.concatenate([
        current_result["thrust"],
        current_result["gravity"],
        current_result["drag"],
        current_result["net"]
    ])

    limit = max(
        5.0,
        np.max(np.abs(all_forces)) * 1.2
    )

    ax_force.set_xlim(-limit, limit)
    ax_force.set_ylim(-limit, limit)


def update(frame):

    global frame_index

    path = current_result["path"]
    times = current_result["times"]

    i = min(
        frame_index,
        len(path) - 1
    )

    # Flight path
    path_line.set_data(
        path[:i + 1, 0],
        path[:i + 1, 1]
    )

    rocket_dot.set_data(
        [path[i, 0]],
        [path[i, 1]]
    )

    # Landing marker
    if i == len(path) - 1:

        landing_dot.set_data(
            [path[-1, 0]],
            [0.0]
        )

    else:
        landing_dot.set_data([], [])

    # Burnout marker
    burnout_index = np.searchsorted(
        times, burn_time
    )

    if (
        i >= burnout_index
        and burnout_index < len(path)
    ):

        burnout_dot.set_data(
            [path[burnout_index, 0]],
            [path[burnout_index, 1]]
        )

    else:
        burnout_dot.set_data([], [])

    # Force vectors
    forces = {
        "Thrust": current_result["thrust"][i],
        "Gravity": current_result["gravity"][i],
        "Drag": current_result["drag"][i],
        "Net": current_result["net"][i]
    }

    for name, vector in forces.items():

        force_arrows[name].set_UVC(
            [vector[0]],
            [vector[1]]
        )

    # Thrust components
    fx, fy = forces["Thrust"]

    fx_line.set_data(
        [0, fx],
        [0, 0]
    )

    fy_line.set_data(
        [fx, fx],
        [0, fy]
    )

    # Numerical values
    pos = path[i]
    vel = current_result["velocities"][i]
    acc = current_result["accelerations"][i]

    actual_angle = current_result["angles"][i]
    command_angle = current_result["commands"][i]

    net = forces["Net"]
    current_time = times[i]

    # Motor status
    if i == len(path) - 1:

        status = "LANDED"
        status_color = "darkorange"

    elif current_time < burn_time:

        remaining = burn_time - current_time

        status = (
            f"MOTOR BURNING | "
            f"{remaining:.2f} s left"
        )

        status_color = "green"

    else:

        status = "MOTOR BURNOUT | COASTING"
        status_color = "red"

    motor_status.set_text(status)
    motor_status.set_color(status_color)

    info_text.set_text(
        f"Time: {current_time:.2f} s\n"
        f"Command angle: {command_angle:.1f} deg\n"
        f"Actual angle:  {actual_angle:.1f} deg\n\n"
        f"Position (m):\n"
        f"  ({pos[0]:.2f}, {pos[1]:.2f})\n"
        f"Velocity (m/s):\n"
        f"  ({vel[0]:.2f}, {vel[1]:.2f})\n"
        f"Acceleration (m/s^2):\n"
        f"  ({acc[0]:.2f}, {acc[1]:.2f})\n"
        f"Net force (N):\n"
        f"  ({net[0]:.2f}, {net[1]:.2f})\n"
        f"Thrust Fx: {fx:.2f} N\n"
        f"Thrust Fy: {fy:.2f} N"
    )

    ax_path.set_title(
        f"{current_name} | "
        f"t = {current_time:.2f} s"
    )

    # Advance animation
    frame_index += animation_step

    if frame_index >= len(path):
        frame_index = 0

    return (
        path_line,
        rocket_dot,
        landing_dot,
        burnout_dot,
        fx_line,
        fy_line,
        info_text,
        motor_status,
        *force_arrows.values()
    )


def change_experiment(label):

    global current_name
    global current_result
    global frame_index

    current_name = label
    current_result = results[label]

    frame_index = 0

    path_line.set_data([], [])
    rocket_dot.set_data([], [])
    landing_dot.set_data([], [])
    burnout_dot.set_data([], [])

    set_limits()
    fig.canvas.draw_idle()


selector.on_clicked(change_experiment)

set_limits()

animation = FuncAnimation(
    fig,
    update,
    interval=30,
    cache_frame_data=False,
    blit=False
)

plt.show()


# --------------------------------------------------
# Parametric position and TVC angle graphs
# --------------------------------------------------

result = results["Direction reversal"]

t = result["times"]
x = result["path"][:, 0]
y = result["path"][:, 1]

commanded = result["commands"]
actual = result["angles"]

fig_param, axes = plt.subplots(
    3, 1,
    figsize=(10, 9),
    sharex=True
)

# Horizontal position x(t)
axes[0].plot(
    t, x,
    color="blue"
)

axes[0].set_ylabel("x(t) (m)")
axes[0].set_title(
    "Parametric Position and Smooth TVC Angle"
)
axes[0].grid()

# Vertical position y(t)
axes[1].plot(
    t, y,
    color="green"
)

axes[1].set_ylabel("y(t) (m)")
axes[1].grid()

# TVC angle
axes[2].step(
    t, commanded,
    where="post",
    linestyle="--",
    color="red",
    label="Commanded Angle"
)

axes[2].plot(
    t, actual,
    color="purple",
    label="Smooth Actual Angle"
)

axes[2].set_ylabel("TVC Angle (deg)")
axes[2].set_xlabel("Time (s)")
axes[2].grid()
axes[2].legend()

# Burnout markers
for ax in axes:

    ax.axvline(
        burn_time,
        color="black",
        linestyle=":",
        label="Motor Burnout"
    )

plt.tight_layout()
plt.show()