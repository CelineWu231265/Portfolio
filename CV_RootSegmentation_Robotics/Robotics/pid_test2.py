"""
PID Controller Evaluation Script
--------------------------------
Evaluates a 3-axis PID controller for the OT-2 pipette simulation. The script
executes a full control loop toward a defined spatial target, records the
position and error evolution, and visualizes system performance.

Author: Celine Wu
Task: PID Control (ILO 8.6B)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from pid import PIDController
from sim_class import Simulation


def evaluate_pid(target_pos, step_time=1.0, steps_limit=1000, tolerance=0.001, stability_window=50):
    """Run a PID controller test toward a specified 3D target position."""
    sim = Simulation(num_agents=1, render=True)
    state = sim.reset(num_agents=1)
    pipette_id = int(list(state.keys())[-1][-1])
    current_pos = np.array(sim.get_pipette_position(pipette_id))

    # Independent controllers for each spatial axis
    controllers = {
        "x": PIDController(10, 0, 0),
        "y": PIDController(10, 0, 0),
        "z": PIDController(10, 0, 0)
    }

    data_log = []
    errors, cmds = [], []
    steady_counter = 0

    print(f"Starting PID evaluation towards target {target_pos.tolist()}")

    for idx in range(steps_limit):
        delta = target_pos - current_pos
        command = np.array([
            controllers["x"].compute(delta[0], step_time),
            controllers["y"].compute(delta[1], step_time),
            controllers["z"].compute(delta[2], step_time)
        ])
        command = np.clip(command, -1, 1)
        sim.run([np.concatenate([command, [0]])])
        current_pos = np.array(sim.get_pipette_position(pipette_id))
        distance = np.linalg.norm(current_pos - target_pos)

        # Data collection
        data_log.append([idx, *current_pos, distance])
        errors.append(delta.tolist())
        cmds.append(command.tolist())
        print(f"[{idx:04d}] Pos={current_pos.round(4)} | Dist={distance:.6f}")

        # Check steady hold condition
        if distance < tolerance:
            steady_counter += 1
            if steady_counter >= stability_window:
                print(f"Target stabilized for {stability_window} consecutive steps.")
                break
        else:
            steady_counter = 0
    else:
        print("PID trial ended without stable convergence.")

    sim.close()
    save_pid_results(data_log)
    plot_pid_performance(np.array(data_log), np.array(errors), np.array(cmds))
    return True


def save_pid_results(data):
    """Write recorded simulation data to a CSV file."""
    header = ["Step", "X", "Y", "Z", "DistanceToTarget"]
    with open("pid_eval_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print("→ Data logged to pid_eval_data.csv")


def plot_pid_performance(positions, errors, commands):
    """Generate and save performance plots."""
    # Position tracking
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 1], label="X-axis")
    plt.plot(positions[:, 0], positions[:, 2], label="Y-axis")
    plt.plot(positions[:, 0], positions[:, 3], label="Z-axis")
    plt.title("PID Position Tracking")
    plt.xlabel("Step")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pid_eval_positions.png")
    plt.show()

    # Distance to target
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 4], color="crimson", label="Distance to Target")
    plt.title("Distance to Target vs Time")
    plt.xlabel("Step")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pid_eval_distance.png")
    plt.show()

    # Error per axis
    plt.figure()
    plt.plot(positions[:, 0], errors[:, 0], label="X Error")
    plt.plot(positions[:, 0], errors[:, 1], label="Y Error")
    plt.plot(positions[:, 0], errors[:, 2], label="Z Error")
    plt.title("PID Axis Error Evolution")
    plt.xlabel("Step")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pid_eval_errors.png")
    plt.show()

    # Velocity commands
    plt.figure()
    plt.plot(positions[:, 0], commands[:, 0], label="Cmd X")
    plt.plot(positions[:, 0], commands[:, 1], label="Cmd Y")
    plt.plot(positions[:, 0], commands[:, 2], label="Cmd Z")
    plt.title("Control Commands Over Time")
    plt.xlabel("Step")
    plt.ylabel("Command Magnitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pid_eval_commands.png")
    plt.show()


if __name__ == "__main__":
    target_position = np.array([0.1, 0.1, 0.25])
    evaluate_pid(target_position)
