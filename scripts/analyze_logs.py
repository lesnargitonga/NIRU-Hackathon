import pandas as pd
import glob
import os
import sys
import re
import numpy as np
from datetime import datetime

def analyze_latest_log():
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
    
    # Check for critical columns
    has_note = 'note' in df.columns

    print(f"Duration:    {duration:.1f}s")
    print(f"Speed (Avg): {avg_speed:.2f} m/s (Max: {max_speed:.2f})")
    print(f"Min Depth:   {min_dmin:.2f}m")
    
    # Analyze Events from "note" column
    print("\n--- Safety Interventions & Events ---")
    
    events = []
    
    # Regex patterns for safety triggers
    patterns = [
        (r'intercept clear=([\d\.]+)', 'HARD INTERCEPT (Path Blocked)'),
        (r'EMERGENCY_STOP', 'EMERGENCY STOP (Dmin < 0.6m)'),
        (r'brake_stop clr=([\d\.]+)', 'Brake to Stop (Clearance < stop_d)'),
        (r'collision_recover', 'COLLISION RECOVER MODE'),
        (r'escape close=1', 'ESCAPE MANEUVER'),
        (r'proactive_climb', 'Proactive Climb'),
        (r'post_coll_freeze', 'Post-Collision Freeze'),
    ]

    last_event_type = None
    event_start_t = None
    event_count = 0

    if has_note:
        for idx, row in df.iterrows():
            note = str(row['note'])
            t = row['t']
            
            current_event = None
            for pat, name in patterns:
                match = re.search(pat, note)
                if match:
                    current_event = name
                    # If it has a value, append it
                    if match.groups():
                        current_event += f" (val={match.group(1)})"
                    break
            
            # Simple debounce/grouping
            if current_event:
                if current_event != last_event_type:
                    rel_t = t - df['t'].iloc[0]
                    events.append(f"T+{rel_t:05.1f}s | {current_event} | Speed={row['speed']:.1f} PathClr={row.get('path_clear', '?'):.1f} Dmin={row.get('min_depth', '?'):.2f}")
                    last_event_type = current_event
                    event_count += 1
            else:
                last_event_type = None

    if not events:
        print("No major safety overrides triggered (Smooth flight?)")
    else:
        # Deduplicate sequential same-events slightly for cleaner log
        for e in events[:25]: # Show first 25
            print(e)
        if len(events) > 25:
            print(f"... and {len(events)-25} more events.")

    # Time-series ASCII visualization
    print("\n--- Flight Profile (Speed vs Clearance) ---")
    print(f"{'Time':<8} | {'Speed (m/s)':<15} | {'Path Clear (m)':<15} | {'Min Depth (m)':<15} | {'Status'}")
    print("-" * 80)
    
    # Downsample for display (every ~1s or 20 rows)
    step = max(1, len(df) // 30)
    for idx in range(0, len(df), step):
        row = df.iloc[idx]
        t = row['t'] - df['t'].iloc[0]
        spd = row['speed']
        clr = row.get('path_clear', 0)
        dmin = row.get('min_depth', 0)
        
        # Visual bars
        spd_bar = '#' * int(spd * 2)
        
        # Color code clearance
        # Just simple text for now
        status = ""
        if 'intercept' in str(row.get('note', '')): status = "[INTERCEPT]"
        elif 'brake' in str(row.get('note', '')): status = "[BRAKING]"
        elif 'escape' in str(row.get('note', '')): status = "[ESCAPE]"
        elif 'collision' in str(row.get('note', '')): status = "[CRASH]"

        print(f"{t:05.1f}s   | {spd:4.1f} {spd_bar:<10} | {clr:5.1f} {'':<9} | {dmin:5.2f} {'':<9} | {status}")


if __name__ == "__main__":
    analyze_latest_log()
