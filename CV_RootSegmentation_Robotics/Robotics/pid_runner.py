import numpy as np
import csv, time, random
from pathlib import Path
from pid import PIDController           # your class
from sim_class import Simulation        # your sim

DT = 1/60.0              # controller timestep (s). Adjust if your sim exposes a dt.
TOL = 0.001              # 1 mm for ILO 8.7D; set 0.01 for 10 mm (C level)
HOLD_STEPS = 30          # must remain within TOL this many steps
MAX_STEPS = 3000         # per target
VEL_CLAMP = 1.0          # action clamp per axis (matches your np.clip([-1,1]))

# PID gains (start with a bit of D to speed settling; you can keep your P-only if you like)
Gx = PIDController(kp=12.0, ki=0.0, kd=0.5)
Gy = PIDController(kp=12.0, ki=0.0, kd=0.5)
Gz = PIDController(kp=14.0, ki=0.0, kd=0.6)

def pick_agent_id(obs_dict):
    # your obs keys looked like "...<id>" — keep the same extraction
    return int(list(obs_dict.keys())[-1][-1])

def load_envelope(csv_path="/Users/celinewu/Documents/GitHub/2024-25b-fai2-adsai-CelineWu231265/Retake/working_envelope.csv"):
    """
    Loads envelope corner points from CSV with or without headers.
    Accepts columns named X,Y,Z (any case) or bare numeric rows.
    Returns (lo, hi) as 3D np.arrays.
    """
    import numpy as np, csv

    # First try: named columns (X,Y,Z ...)
    try:
        arr_named = np.genfromtxt(
            csv_path, delimiter=",", names=True, dtype=float, encoding=None
        )
        if arr_named.dtype.names:  # we had a header
            # map any case 'x','y','z'
            names_lower = [n.lower() for n in arr_named.dtype.names]
            def col(name):
                i = names_lower.index(name)
                return np.asarray(arr_named[arr_named.dtype.names[i]], dtype=float)
            if all(n in names_lower for n in ("x","y","z")):
                xyz = np.column_stack([col("x"), col("y"), col("z")])
                lo = xyz.min(axis=0)
                hi = xyz.max(axis=0)
                return lo, hi
    except Exception:
        pass

    # Fallback: plain numeric rows, skip non-numeric lines safely
    pts = []
    with open(csv_path, newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            cells = [c.strip() for c in row if c.strip() != ""]
            if len(cells) < 3:
                continue
            try:
                x, y, z = map(float, cells[:3])
                pts.append([x, y, z])
            except ValueError:
                # probably a header or junk line — skip
                continue

    if len(pts) < 4:
        raise ValueError("working_envelope.csv must contain ≥ 4 numeric rows of (X,Y,Z).")

    xyz = np.asarray(pts, dtype=float)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    return lo, hi


def random_targets(lo, hi, n=6, margin=0.005):
    lo2 = lo + margin; hi2 = hi - margin
    return [np.array([random.uniform(lo2[i], hi2[i]) for i in range(3)]) for _ in range(n)]

def go_to(sim, agent_id, target):
    # reset PIDs each move
    Gx.reset(); Gy.reset(); Gz.reset()
    hold = 0
    t0 = time.time()
    best_err = float('inf')

    for step in range(MAX_STEPS):
        cur = np.array(sim.get_pipette_position(agent_id), dtype=float)
        err = target - cur
        best_err = min(best_err, float(np.linalg.norm(err)))

        vx = Gx.compute(err[0], DT)
        vy = Gy.compute(err[1], DT)
        vz = Gz.compute(err[2], DT)

        ctrl = np.clip([vx, vy, vz], -VEL_CLAMP, VEL_CLAMP)
        action = np.concatenate([ctrl, [0.0]])   # dummy gripper

        sim.run([action])                         # advance sim 1 control step
        time.sleep(DT)

        dist = np.linalg.norm(err)
        if dist < TOL:
            hold += 1
            if hold >= HOLD_STEPS:
                return True, time.time()-t0, dist, best_err
        else:
            hold = 0
    return False, time.time()-t0, dist, best_err

def main():
    lo, hi = load_envelope("working_envelope.csv")
    targets = random_targets(lo, hi, n=8)

    sim = Simulation(num_agents=1, render=True)
    obs = sim.reset(num_agents=1)
    agent_id = pick_agent_id(obs)

    results = []
    for i, tgt in enumerate(targets, 1):
        ok, elapsed, final_err, best_err = go_to(sim, agent_id, tgt)
        print(f"[{i}] target={tgt} | {'OK' if ok else 'FAIL'} | "
              f"time={elapsed:.2f}s | final_err={final_err*1000:.2f} mm | best_err={best_err*1000:.2f} mm")
        results.append((ok, elapsed, final_err, best_err))

    sim.close()

    # Summary
    oks = [r[0] for r in results]
    worst_final = max(r[2] for r in results)
    avg_time = sum(r[1] for r in results)/len(results)
    print("\n=== PID Summary ===")
    print(f"All targets OK: {all(oks)}")
    print(f"Worst final error: {worst_final*1000:.2f} mm (threshold {TOL*1000:.1f} mm)")
    print(f"Average settle time: {avg_time:.2f} s")

    # CSV log for evidencing
    with open("task12_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","ok","time_s","final_err_m","best_err_m","target_x","target_y","target_z"])
        for i,(ok,t,fe,be),tgt in zip(range(1,len(results)+1), results, targets):
            w.writerow([i,int(ok),f"{t:.4f}",f"{fe:.6f}",f"{be:.6f}",*map(lambda x:f"{x:.6f}", tgt)])

if __name__ == "__main__":
    main()
