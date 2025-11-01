import numpy as np
import time
import matplotlib.pyplot as plt
from pid import PIDController
from sim_class import Simulation
import pybullet as p


def safe_run(sim, action):
    """Safely run the simulation and reconnect if the physics server disconnects."""
    try:
        sim.run([action])
    except p.error:
        print("Reconnecting to physics server...")
        sim.close()
        sim = Simulation(num_agents=1, render=True)
        sim.reset(num_agents=1)
    return sim


def move_to_target(sim, agent_id, target, pid_x, pid_y, pid_z,
                   dt=1/25.0, max_steps=900, threshold=0.001, hold_steps=60):
    position = np.array(sim.get_pipette_position(agent_id))
    hold_counter = 0
    trajectory = []

    print(f"Moving to target {np.round(target, 3)}")

    for step in range(max_steps):
        error = target - position
        vx = pid_x.compute(error[0], dt)
        vy = pid_y.compute(error[1], dt)
        vz = pid_z.compute(error[2], dt)
        control = np.clip([vx, vy, vz], -1.0, 1.0)

        sim = safe_run(sim, np.concatenate([control, [0]]))
        position = np.array(sim.get_pipette_position(agent_id))
        distance = np.linalg.norm(position - target)
        trajectory.append([step, *position, distance])

        if step % 10 == 0:
            print(f"Step {step:03d} | Distance={distance*1000:.1f} mm")


        if len(trajectory) > 50 and abs(distance - trajectory[-50][4]) < 1e-5:
            print("Motion stalled, resetting PID.")
            pid_x.reset()
            pid_y.reset()
            pid_z.reset()
            break

        if distance < threshold:
            hold_counter += 1
            if hold_counter >= hold_steps:
                print(f"Target reached within {threshold*1000:.1f} mm.")
                break
        else:
            hold_counter = 0

    return np.array(trajectory), sim


def run_pid_visualization():
    sim = Simulation(num_agents=1, render=True)
    obs = sim.reset(num_agents=1)
    agent_id = int(list(obs.keys())[-1][-1])  # correct agent ID


    # PID tuning parameters
    Kp, Ki, Kd = 20.0, 0.05, 0.6
    pid_x = PIDController(Kp, Ki, Kd)
    pid_y = PIDController(Kp, Ki, Kd)
    pid_z = PIDController(Kp, Ki, Kd)

    # Large movements across the workspace
    targets = [
        np.array([0.08, 0.08, 0.18]),
        np.array([0.14, 0.14, 0.27]),
        np.array([0.08, 0.16, 0.25]),
        np.array([0.16, 0.08, 0.20]),
        np.array([0.10, 0.10, 0.25])
    ]

    all_trajs = []

    for loop in range(2):
        print(f"Loop {loop + 1}/2")
        for t in targets:
            traj, sim = move_to_target(sim, agent_id, t, pid_x, pid_y, pid_z)
            all_trajs.append(traj)
            time.sleep(1.5)

    print("Simulation complete, holding scene for capture...")

    sim.close()

    # Save distance plot
    combined = np.vstack(all_trajs)
    plt.figure()
    plt.plot(combined[:, 0], combined[:, 4] * 1000, color="blue")
    plt.title("PID Controller Multi-Target Path")
    plt.xlabel("Step")
    plt.ylabel("Distance to Target (mm)")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("pid_visualization_distance.png")
    plt.close()
    print("Saved plot: pid_visualization_distance.png")


if __name__ == "__main__":
    run_pid_visualization()
