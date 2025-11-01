"""
PID Controller Tuning Experiment
--------------------------------
This script systematically tests multiple PID gain combinations for the OT-2 pipette simulation.
Each run evaluates position accuracy, convergence speed, and steady-state error.

Author: Celine Wu
Task: PID Controller Tuning (ILO 8.6B)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from pid import PIDController
from sim_class import Simulation


def run_pid_trial(target, gains, dt=1.0, max_steps=800, threshold=0.001, hold_steps=30):
    """Run a single PID trial and return final error and settle time."""
    Kp, Ki, Kd = gains
    sim = Simulation(num_agents=1, render=False)
    obs = sim.reset(num_agents=1)
    agent_id = int(list(obs.keys())[-1][-1])
    position = np.array(sim.get_pipette_position(agent_id))

    # Create controllers for each axis
    ctrl_x, ctrl_y, ctrl_z = PIDController(Kp, Ki, Kd), PIDController(Kp, Ki, Kd), PIDController(Kp, Ki, Kd)

    hold_count = 0
    trajectory = []

    start_time = time.time()

    for step in range(max_steps):
        error = target - position
        velocity = np.array([
            ctrl_x.compute(error[0], dt),
            ctrl_y.compute(error[1], dt),
            ctrl_z.compute(error[2], dt),
        ])
        velocity = np.clip(velocity, -1.0, 1.0)
        sim.run([np.concatenate([velocity, [0]])])
        position = np.array(sim.get_pipette_position(agent_id))
        distance = np.linalg.norm(position - target)
        trajectory.append([step, *position, distance])

        if distance < threshold:
            hold_count += 1
            if hold_count >= hold_steps:
                break
        else:
            hold_count = 0

    runtime = time.time() - start_time
    sim.close()
    final_err = float(np.linalg.norm(position - target))
    settle_time = runtime
    return final_err, settle_time, np.array(trajectory)


def tune_pid():
    """Test multiple PID gain sets and save comparative results."""
    target = np.array([0.1, 0.1, 0.25])
    gain_sets = [
        (5, 0, 0),
        (10, 0, 0),
        (10, 0.2, 0.5),
        (12, 0.1, 0.3),
        (15, 0, 0.6),
        (8, 0.05, 0.4),
    ]

    results = []

    for gains in gain_sets:
        print(f"\n--- Testing Kp={gains[0]}, Ki={gains[1]}, Kd={gains[2]} ---")
        error, settle, trajectory = run_pid_trial(target, gains)
        results.append([*gains, error, settle])
        np.savetxt(f"pid_tune_traj_Kp{gains[0]}_Ki{gains[1]}_Kd{gains[2]}.csv", trajectory, delimiter=",")
        print(f"Final error: {error*1000:.3f} mm | Settle time: {settle:.2f} s")

        # Plot each run's distance curve
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 4], label="Distance to Target")
        plt.title(f"PID Tuning Trial Kp={gains[0]}, Ki={gains[1]}, Kd={gains[2]}")
        plt.xlabel("Step")
        plt.ylabel("Distance (m)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pid_tuning_Kp{gains[0]}_Ki{gains[1]}_Kd{gains[2]}.png")
        plt.close()

    # Save all test results
    with open("pid_tuning_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Kp", "Ki", "Kd", "FinalError(m)", "SettleTime(s)"])
        writer.writerows(results)

    # Display summary
    print("\n=== PID Tuning Summary ===")
    for row in results:
        print(f"Kp={row[0]:.2f}, Ki={row[1]:.2f}, Kd={row[2]:.2f} | Final Error={row[3]*1000:.3f} mm | Time={row[4]:.2f}s")

    # Highlight best configuration
    best = min(results, key=lambda r: r[3])
    print(f"\nBest configuration: Kp={best[0]}, Ki={best[1]}, Kd={best[2]} (Error={best[3]*1000:.3f} mm)")


if __name__ == "__main__":
    tune_pid()
