from sim_class import Simulation
import csv, json, math

VEL_MAG = 0.5      
EPS = 1e-4         
PATIENCE = 20       
MAX_STEPS = 2000    

def tip_pos(state, sim):
    key = f"robotId_{sim.robotIds[0]}"
    return tuple(map(float, state[key]["pipette_position"]))

def move_until_stop(sim, vx, vy, vz, max_steps=MAX_STEPS, eps=EPS, patience=PATIENCE):
    actions = [[vx, vy, vz, 0]]  # drop=0
    prev = None
    stable = 0
    state = None
    for _ in range(max_steps):
        state = sim.run(actions)
        curr = tip_pos(state, sim)
        if prev is not None:
            if all(abs(c - p) < eps for c, p in zip(curr, prev)):
                stable += 1
                if stable >= patience:
                    break
            else:
                stable = 0
        prev = curr
    return state

def main():
    sim = Simulation(num_agents=1)

    # Corner-visit pattern (friend’s sequence)
    corner_velocities = [
        (-VEL_MAG, -VEL_MAG, 0.0),  # Bottom-left-front
        ( VEL_MAG,  0.0,     0.0),  # Bottom-right-front
        ( 0.0,      VEL_MAG, 0.0),  # Bottom-right-back
        (-VEL_MAG,  0.0,     0.0),  # Bottom-left-back
        ( 0.0,      0.0,     VEL_MAG),  # Up to top-left-back
        ( 0.0,     -VEL_MAG, 0.0),  # Top-left-front
        ( VEL_MAG,  0.0,     0.0),  # Top-right-front
        ( 0.0,      VEL_MAG, 0.0),  # Top-right-back
    ]

    working_envelope = []
    for vx, vy, vz in corner_velocities:
        state = move_until_stop(sim, vx, vy, vz)
        working_envelope.append(tip_pos(state, sim))

    # Save CSV
    with open("working_envelope.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "Z"])
        w.writerows(working_envelope)

    # Save JSON (optional but handy)
    with open("working_envelope.json", "w") as f:
        json.dump({"corners": working_envelope}, f, indent=2)

    # Print results
    print("Working Envelope Coordinates:")
    for p in working_envelope:
        print(p)
    print("Saved: working_envelope.csv, working_envelope.json")

if __name__ == "__main__":
    main()