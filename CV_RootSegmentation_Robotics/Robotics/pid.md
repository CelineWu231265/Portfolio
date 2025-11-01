# OT-2 PID Controller Simulation

This project implements a **PID (Proportional–Integral–Derivative)** controller for the **Opentrons OT-2** robotic arm inside a simulated environment using **PyBullet**.  
The controller enables precise motion of the pipette tip toward target coordinates, demonstrating closed-loop control and performance evaluation using classical control theory.

---

## 📦 Project Overview

The goal of this project is to move the pipette tip of the OT-2 to defined 3D positions with high precision.  
Three separate PID controllers are used, one for each axis (**X**, **Y**, and **Z**).  
Each controller continuously computes the error between the current and desired position and adjusts the movement velocity accordingly.

After tuning, the system achieves sub-millimetre accuracy with smooth convergence to targets across the working envelope.

---

## ⚙️ Setup and Dependencies

### Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib pybullet
```

---

## ▶️ Running the Simulation

To test the PID controller on a single target and generate performance plots:

```bash
python pid_test.py
```

To tune and compare multiple PID configurations automatically:

```bash
python pid_test1.py
```

To run a multi-target visualization (for visible motion between positions):

```bash
python pid_visualize.py
```

Simulation logs and plots are automatically saved as `.csv` and `.png` files in the working directory.

---

## 📁 File Descriptions

| File | Description |
|------|--------------|
| `pid.py` | Defines the `PIDController` class for X, Y, and Z control. |
| `pid_test.py` | Runs a single-target test and logs results (position, distance, error). |
| `pid_test1.py` | Performs multiple tuning tests with different PID gains and reports performance metrics. |
| `pid_test2.py` | Alternative test configuration for extended PID testing. |
| `pid_visualize.py` | Runs the controller across several distant targets for visual demonstration and GIF creation. |
| `pid_results.csv` | Stores position and distance to target for each step of a PID run. |
| `pid_position.png` | Position along X, Y, Z vs. step count. |
| `pid_distance.png` | Distance to target vs. time (convergence graph). |
| `pid_errors.png` | Per-axis error evolution over time. |
| `pid_velocities.png` | Control velocities (PID outputs) for each axis. |
| `working_envelope.csv` | Defines the reachable corners of the OT-2 pipette’s workspace. |
| `sim_class.py` | Handles environment setup, simulation control, and pipette feedback. |
| `pid.mp4` / `pid.gif` | Visual demonstration of the robot reaching the target positions. |

---

## 🧠 PID Control Logic

Each axis of motion is independently controlled by a PID loop of the form:

```
u(t) = Kp * e(t) + Ki * ∫ e(t) dt + Kd * de(t)/dt
```

Where:

- **e(t)** = difference between target and current position  
- **Kp** = proportional gain (immediate correction)  
- **Ki** = integral gain (long-term error compensation)  
- **Kd** = derivative gain (damping and prediction)  

The control outputs are applied as velocity commands and clipped within a safe range to prevent unstable oscillations.

---

## 🔬 Performance Evaluation

The system performance is evaluated by logging the distance to the target and settling time for each PID parameter configuration.  
Data is saved to `pid_results.csv` and visualized using **matplotlib**.

### Example tuning results:

| Kp | Ki | Kd | Final Error (mm) | Settling Time (s) |
|----|----|----|------------------|-------------------|
| 5.0 | 0.0 | 0.0 | 0.529 | 0.03 |
| 10.0 | 0.0 | 0.0 | 0.281 | 0.01 |
| 10.0 | 0.2 | 0.5 | 0.277 | 0.02 |
| 12.0 | 0.1 | 0.3 | 0.736 | 0.03 |
| 15.0 | 0.0 | 0.6 | 0.145 | 0.01 |

✅ **Best performing configuration:**

```
Kp = 15.0, Ki = 0.0, Kd = 0.6
Final positioning error: 0.145 mm (meets <1 mm accuracy goal)
```

---

## 🧩 Simulation Behaviour

- The simulation opens a **PyBullet GUI** window displaying the OT-2 robot.  
- Velocity commands are sent continuously until the target is reached.  
- The motion is smoothed and stabilized by PID control.  
- Once within a threshold (1 mm), the controller holds the position for a number of steps (`hold_steps`) before moving to the next target.

### Example Targets Used

```python
targets = [
    [0.05, 0.05, 0.15],
    [0.17, 0.17, 0.30],
    [0.05, 0.17, 0.28],
    [0.17, 0.05, 0.18],
    [0.10, 0.10, 0.25]
]
```

This creates visible, large-scale movements across the workspace.

---

## 📊 Plots and Visual Outputs

After each run, the following plots are generated automatically:

- `pid_position.png` – Position values for X, Y, Z over time  
- `pid_distance.png` – Distance to target showing convergence  
- `pid_errors.png` – Per-axis error decay over steps  
- `pid_velocities.png` – Controller output for each axis  

A video recording (`pid.mp4`) or GIF (`pid.gif`) can be created to visually demonstrate the pipette reaching its targets.

### To create a GIF from a recorded video:

```bash
ffmpeg -i pid.mp4 -vf "fps=20,scale=640:-1:flags=lanczos" pid.gif
```

---

## 🧾 Working Envelope

To confirm the robot’s reachable area, the pipette was moved to each corner of the workspace, and its coordinates were logged.  
The boundaries were saved in `working_envelope.csv`.  
This provides situational awareness and ensures that all target positions lie within the OT-2’s physical limits.

---

## 🧠 Key Features

- Classical **PID controller** (3-axis implementation)  
- Real-time simulation and feedback in **PyBullet**  
- Automatic performance logging and plotting  
- Accurate **sub-millimetre positioning (<1 mm)**  
- Repeatable and fully documented codebase  

---

## 🚀 Future Improvements

- Integrate a camera-based perception system for automatic target coordinate detection  
- Introduce adaptive or reinforcement learning-based tuning of PID parameters  
- Extend to multi-agent or dual-arm coordination  
- Implement trajectory smoothing and motion planning for continuous path control  

---

## 👩‍💻 Author

**Celine Wu**  
*Applied Data Science for Artificial Intelligence*  
**Breda University of Applied Sciences – 2024/2025**
