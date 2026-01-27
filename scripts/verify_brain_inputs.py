import numpy as np
import sys
import os

# Add parent directory to path so we can import rl modules if needed, 
# but for this test we might just want to simulate the vector construction 
# or import the env if possible.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rl"))

print("--- LESNAR BRAIN PRE-FLIGHT CHECK ---")

def verify_brain_inputs(obs_vector):
    # Obs vector: [Vision(0), State(1-19)...]
    # We probably won't have vision here if we just test the vector logic
    # But let's assume we are passing the kinematic vector directly.
    
    print(f"Vector Shape: {obs_vector.shape} (Expected: (19,))")
    
    if obs_vector.shape[0] != 19:
        print("CRITICAL FAILURE: Vector shape is wrong. Did you update airsim_gym_env.py?")
        return

    # Indices based on the new code:
    # 0-2: Vel
    # 3-5: Att
    # 6: Alt
    # 7: BodyX
    # 8: BodyY
    # 9: Dist
    # 10: Sin(Head)
    # 11: Cos(Head)
    # 12: Track
    # 13: TTC
    # 14: Eff
    # 15-18: Proprioception

    # Check 1: Relative Coordinates
    dist = obs_vector[9] 
    body_x = obs_vector[7]
    print(f"Distance Input: {dist:.4f} (Should be relative/normalized)")
    print(f"Body X Input: {body_x:.4f}")
    
    if abs(dist) > 5.0: # 5.0 * 50 = 250m. Unlikely to be that far.
        print("WARNING: Distance input is very high. Is it normalized?")

    # Check 2: Heading Continuity (Sin/Cos)
    sin_h = obs_vector[10]
    cos_h = obs_vector[11]
    magnitude = np.sqrt(sin_h**2 + cos_h**2)
    print(f"Heading Inputs: Sin={sin_h:.4f}, Cos={cos_h:.4f}")
    
    if abs(magnitude - 1.0) > 0.1:
        print(f"CRITICAL WARNING: Sin/Cos magnitude is {magnitude:.4f}. Should be ~1.0.")
    else:
        print("Heading Input: Continuous (Safe)")

    # Check 3: TTC Clamping
    ttc = obs_vector[13]
    if ttc > 1.05:
        print(f"CRITICAL WARNING: TTC input is {ttc:.4f} (> 1.0). CLAMP IT or gradients will explode.")
    elif ttc < 0.0:
        print(f"CRITICAL WARNING: TTC is negative? {ttc}")
    else:
        print(f"TTC Input: {ttc:.4f} (Safe)")

    print("--- CHECK COMPLETE ---")

# Mock inputs simulating an "infinity" case or "flip" case
print("\nTest Case 1: Open field flight (TTC Infinity)")
mock_ttc_raw = 9999.0
normalized_ttc = min(mock_ttc_raw, 5.0) / 5.0
mock_heading = 3.14159 # 180 degrees
mock_vec = np.zeros(19)
mock_vec[9] = 0.5 # Dist
mock_vec[10] = np.sin(mock_heading)
mock_vec[11] = np.cos(mock_heading)
mock_vec[13] = normalized_ttc

verify_brain_inputs(mock_vec)
