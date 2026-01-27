import pandas as pd
import glob
import os
import sys
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def analyze_and_plot_latest_log():
    # Find latest log
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    list_of_files = glob.glob(os.path.join(log_dir, 'brain_*.csv'))
    if not list_of_files:
        print("No logs found in", log_dir)
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"\nAnalyzing Log: {os.path.basename(latest_file)}")
    print("=" * 60)

    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        print(f"Error reading log: {e}")
        return

    if df.empty:
        print("Log is empty.")
        return

    # Basic Stats
    duration = df['t'].iloc[-1] - df['t'].iloc[0]
    avg_speed = df['speed'].mean()
    max_speed = df['speed'].max()
    min_dmin = df['min_depth'].min() if 'min_depth' in df.columns else np.nan

    print(f"Duration:    {duration:.1f}s")
    print(f"Speed (Avg): {avg_speed:.2f} m/s (Max: {max_speed:.2f})")
    print(f"Min Depth:   {min_dmin:.2f}m")
    
    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot Trajectory (Segmented by Speed)
    # Convert to standard NED coordinates for plotting if needed, generally AirSim is NED.
    # X = North, Y = East.
    # We plot Y (East) on X-axis, X (North) on Y-axis to match typical map orientation if desired, 
    # but standard X-Y plot is fine.
    
    x = df['pos_x'].values
    y = df['pos_y'].values
    speed = df['speed'].values
    
    points = np.array([y, x]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(0, 5.0) # Speed 0-5 m/s
    lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.7)
    lc.set_array(speed)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    
    # 2. Plot Events
    # Regex patterns for safety triggers
    patterns = [
        (r'intercept clear=([\d\.]+)', 'orange', 'X', 'Intercept'),
        (r'EMERGENCY_STOP', 'red', 'P', 'Emg Stop'),
        (r'brake_stop', 'magenta', '*', 'Brake Stop'),
        (r'collision_recover', 'black', 'o', 'Collision Clean'),
        (r'post_coll_freeze', 'brown', 's', 'Post-Coll Freeze'), 
        (r'collision_recover obj=', 'red', 'X', 'Collision Hit') # Prioritize specific object hit
    ]

    # Helper to calculate detected obstacle position
    # obstacle_pos = drone_pos + clear_dist * direction_vector
    # We use 'yaw' (degrees). NED frame: 0 is North (X+), 90 East (Y+)
    # x_new = x + d * cos(yaw), y_new = y + d * sin(yaw)
    
    obs_x = []
    obs_y = []

    # Map events
    legend_handles = {}
    
    print("\n--- Event Log ---")
    
    last_evt_name = None
    for idx, row in df.iterrows():
        note = str(row.get('note', ''))
        # Plot projected obstacle if path is blocked
        # We look for notes like "clear=1.2" or just use path_clear column
        
        path_clear = row.get('path_clear', 100)
        yaw_rad = math.radians(row['steer_deg'] + 0) # This is actually steer, not yaw. Use 'yaw' if available? 
        # Wait, the CSV has 'steer' (steering command) and 'yaw_rate'.
        # We heavily rely on accurate yaw. The CSV usually doesn't dump absolute yaw!
        # Ah, looking at `lesnar_brain.py`, it logs `x, y, z`. But `yaw` is not explicitly in the CSV columns list!
        # It logs: t,source,cls,steer_deg,speed,yaw_rate,path_clear,vfh_clear,primary_angle,vfh_steer,blocked,in_escape,in_collision_recover,min_depth,pos_x,pos_y,pos_z,note
        
        # WE ARE MISSING ABSOLUTE YAW IN THE CSV. 
        # We can approximate it by integrating yaw_rate * dt, but it will drift.
        # OR we assume the drone moves roughly in the direction of velocity (dx, dy).
        
        # Let's estimate direction from position diffs
        if idx < len(df)-1:
            dx = df.iloc[idx+1]['pos_x'] - row['pos_x']
            dy = df.iloc[idx+1]['pos_y'] - row['pos_y']
            heading = math.atan2(dy, dx) # math.atan2(y, x) -> standard math: y is North-South? No.
            # AirSim NED: X=North, Y=East.
            # Grid plot: X axis = Y(East), Y axis = X(North).
            # So plotting (y, x). 
            # Heading vector: (dy, dx).
            
            # If path_clear < 5m, plot an obstacle dot
            if path_clear < 5.0 and path_clear > 0.1:
                # Projected obstacle position
                ox = row['pos_x'] + path_clear * math.cos(heading)
                oy = row['pos_y'] + path_clear * math.sin(heading)
                obs_x.append(oy) # Plot Y on X-axis
                obs_y.append(ox) # Plot X on Y-axis
        
        # Check patterns
        for pat, color, marker, label in patterns:
            if re.search(pat, note):
                # Debounce print
                if label != last_evt_name:
                    print(f"T+{row['t'] - df['t'].iloc[0]:.1f}s: {label} @ [{row['pos_x']:.1f}, {row['pos_y']:.1f}] Note: {note}")
                    last_evt_name = label
                
                scatter = ax.scatter(row['pos_y'], row['pos_x'], c=color, marker=marker, s=100, zorder=5, label=label if label not in legend_handles else "")
                if label not in legend_handles:
                    legend_handles[label] = scatter
                break
    
    # Plot obstacles
    if obs_x:
        ax.scatter(obs_x, obs_y, c='gray', s=10, alpha=0.5, label='Detected Obstacles', marker='.')
        
    ax.set_aspect('equal')
    ax.set_title(f'Drone Trajectory & Events\nLog: {os.path.basename(latest_file)}', fontsize=10)
    ax.set_xlabel('East (Y) [m]')
    ax.set_ylabel('North (X) [m]')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Create legend from unique handles
    # handles, labels = ax.get_legend_handles_labels() # This often duplicates or misses
    # Use our manual dict
    if legend_handles:
        ax.legend(legend_handles.values(), legend_handles.keys(), loc='best')
    
    # Colorbar
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Speed (m/s)')
    
    # Save
    out_path = latest_file.replace('.csv', '_viz.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to: {out_path}")
    print("\nSummary:")
    print("1. BLUE/PURPLE lines are slow, YELLOW is fast.")
    print("2. RED markers are stops/collisions.")
    print("3. GRAY dots are where the drone 'thought' obstacles were.")
    print("   (Note: Obstacle positions attemptedly reconstructed from trajectory heading)")

if __name__ == "__main__":
    analyze_and_plot_latest_log()
