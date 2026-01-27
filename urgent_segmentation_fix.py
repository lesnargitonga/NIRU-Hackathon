"""
URGENT: Segmentation System Diagnostic and Fix
Based on analysis showing 0% gap detection rate across all algorithms
"""

import csv
import json
from pathlib import Path
import math

def diagnose_segmentation_issue():
    """
    Detailed diagnosis of why gap detection is failing
    """
    print("URGENT SEGMENTATION ISSUE DIAGNOSIS")
    print("=" * 45)
    print()
    
    logs_dir = Path("logs")
    
    # Check one of the log files in detail
    commit_file = logs_dir / "seg_run_commit.csv"
    
    if commit_file.exists():
        print("Examining seg_run_commit.csv in detail...")
        
        with open(commit_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"Total records: {len(rows)}")
        print()
        
        # Check first few records
        print("First 5 records:")
        for i, row in enumerate(rows[:5]):
            print(f"  Record {i+1}:")
            for key, value in row.items():
                print(f"    {key}: {value}")
            print()
        
        # Analyze the pattern
        print("Data Pattern Analysis:")
        
        # Check if all values are the same
        first_row = rows[0]
        all_same = True
        for row in rows[1:]:
            for key in first_row:
                if key != 't' and row[key] != first_row[key]:
                    all_same = False
                    break
            if not all_same:
                break
        
        print(f"All non-timestamp values identical: {all_same}")
        
        # Check obstacle detection
        l_values = [float(row['L']) for row in rows]
        c_values = [float(row['C']) for row in rows]
        r_values = [float(row['R']) for row in rows]
        
        print(f"L (Left) sensor values: all {l_values[0]}" if all(v == l_values[0] for v in l_values) else "L values vary")
        print(f"C (Center) sensor values: all {c_values[0]}" if all(v == c_values[0] for v in c_values) else "C values vary")
        print(f"R (Right) sensor values: all {r_values[0]}" if all(v == r_values[0] for v in r_values) else "R values vary")
        
        print()
        print("DIAGNOSIS:")
        if all(v == 1.0 for v in l_values + c_values + r_values):
            print("❌ CRITICAL ISSUE: All obstacle sensors reading 1.0")
            print("   This indicates:")
            print("   1. No obstacles are being detected (all clear)")
            print("   2. Gap detection algorithm expects obstacles to find gaps")
            print("   3. With no obstacles, no gaps can be computed")
            print()
            print("🔧 IMMEDIATE FIXES NEEDED:")
            print("1. CHECK SENSOR CONFIGURATION:")
            print("   - Verify depth/LiDAR sensors are working")
            print("   - Check sensor data is being processed correctly")
            print("   - Ensure obstacle detection thresholds are appropriate")
            print()
            print("2. TEST IN OBSTACLE-RICH ENVIRONMENT:")
            print("   - Move drone to environment with obstacles")
            print("   - Test with walls, trees, or other barriers")
            print("   - Verify sensors can detect these obstacles")
            print()
            print("3. ALGORITHM DEBUGGING:")
            print("   - Add debug logging to segmentation algorithms")
            print("   - Print intermediate processing steps")
            print("   - Verify input data format and units")
        
        # Check area fraction
        area_frac = float(first_row['largest_area_frac'])
        print(f"4. AREA FRACTION ANALYSIS:")
        print(f"   - Largest area fraction: {area_frac:.3f}")
        if area_frac > 0.95:
            print("   - This suggests the entire view is classified as obstacle-free")
            print("   - Gap detection needs obstacles to define gaps")
            print("   - Consider testing in more complex environments")

def create_debug_segmentation_script():
    """Create a debugging script for segmentation system"""
    debug_script = '''
"""
Debug script for segmentation system
Run this to diagnose and fix segmentation issues
"""

import sys
import os
from pathlib import Path

# Add AirSim path if needed
sys.path.append('airsim')

def test_sensor_data():
    """Test if sensors are providing meaningful data"""
    print("Testing sensor data sources...")
    
    # Check if we can import AirSim
    try:
        import airsim
        print("✓ AirSim imported successfully")
        
        # Try to connect to AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ Connected to AirSim")
        
        # Get sensor data
        state = client.getMultirotorState()
        print(f"✓ Drone state retrieved: {state.kinematics_estimated.position}")
        
        # Test depth camera
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
        ])
        
        if responses and len(responses[0].image_data_float) > 0:
            depth_data = responses[0].image_data_float
            min_depth = min(depth_data)
            max_depth = max(depth_data)
            print(f"✓ Depth data: min={min_depth:.2f}m, max={max_depth:.2f}m")
            
            # Check if we're seeing obstacles
            close_pixels = sum(1 for d in depth_data if d < 5.0)  # Within 5 meters
            total_pixels = len(depth_data)
            obstacle_ratio = close_pixels / total_pixels
            
            print(f"✓ Obstacle ratio: {obstacle_ratio:.2%}")
            
            if obstacle_ratio < 0.1:
                print("⚠️  WARNING: Very few obstacles detected")
                print("   - Move to environment with walls/obstacles")
                print("   - Check if drone is too high")
                print("   - Verify depth camera configuration")
        else:
            print("❌ No depth data received")
            
    except ImportError:
        print("❌ Cannot import airsim - check installation")
    except Exception as e:
        print(f"❌ Error testing sensors: {e}")

def test_segmentation_algorithm():
    """Test segmentation algorithm with sample data"""
    print("\\nTesting segmentation algorithms...")
    
    # Import segmentation modules
    try:
        sys.path.append('ai_modules')
        from segmentation_inference import run_segmentation
        print("✓ Segmentation module imported")
        
        # Test with sample obstacle data
        print("Testing with sample obstacle configuration...")
        
        # Simulate obstacle sensor data
        test_cases = [
            {"L": 0.5, "C": 1.0, "R": 0.5, "description": "Gap in center"},
            {"L": 1.0, "C": 0.5, "R": 1.0, "description": "Obstacle in center"},
            {"L": 0.3, "C": 0.8, "R": 1.0, "description": "Left obstacle, right clear"},
            {"L": 1.0, "C": 1.0, "R": 1.0, "description": "All clear (current issue)"}
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"  Test {i+1}: {test_case['description']}")
            print(f"    L={test_case['L']}, C={test_case['C']}, R={test_case['R']}")
            
            # Here you would call your actual segmentation algorithm
            # result = run_segmentation(test_case['L'], test_case['C'], test_case['R'])
            # print(f"    Result: {result}")
            
    except ImportError as e:
        print(f"❌ Cannot import segmentation module: {e}")

def recommend_immediate_actions():
    """Provide immediate action recommendations"""
    print("\\n" + "=" * 50)
    print("IMMEDIATE ACTION PLAN")
    print("=" * 50)
    
    print("\\n1. VERIFY ENVIRONMENT (Do this first!)")
    print("   □ Move drone to environment with obstacles")
    print("   □ Test near walls, trees, or buildings") 
    print("   □ Ensure drone is at appropriate altitude (2-10m)")
    
    print("\\n2. CHECK SENSOR CONFIGURATION")
    print("   □ Verify depth camera is enabled in AirSim settings")
    print("   □ Check sensor update rates")
    print("   □ Confirm obstacle detection thresholds")
    
    print("\\n3. DEBUG SEGMENTATION ALGORITHMS")
    print("   □ Add debug prints to segmentation_inference.py")
    print("   □ Log intermediate processing steps")
    print("   □ Test algorithms with known obstacle patterns")
    
    print("\\n4. TEST INCREMENTAL IMPROVEMENTS")
    print("   □ Start with simple obstacle detection")
    print("   □ Gradually increase complexity")
    print("   □ Validate each step before proceeding")
    
    print("\\n5. MONITORING AND VALIDATION")
    print("   □ Set up real-time visualization")
    print("   □ Create automated testing framework")
    print("   □ Implement performance benchmarks")

if __name__ == "__main__":
    test_sensor_data()
    test_segmentation_algorithm()
    recommend_immediate_actions()
'''
    
    debug_file = Path("debug_segmentation.py")
    with open(debug_file, 'w') as f:
        f.write(debug_script)
    
    print(f"✓ Created debug script: {debug_file}")
    print("  Run this script to diagnose sensor and algorithm issues")

def create_immediate_action_plan():
    """Create step-by-step action plan"""
    action_plan = {
        "immediate_priority": "CRITICAL - 0% gap detection rate",
        "root_cause": "No obstacles detected (all sensors reading 1.0)",
        "immediate_actions": [
            {
                "action": "Test in obstacle-rich environment",
                "priority": "URGENT",
                "time_estimate": "30 minutes",
                "steps": [
                    "Move drone to indoor environment with walls",
                    "Or add obstacles to current simulation environment",
                    "Verify depth sensors detect these obstacles",
                    "Re-run segmentation algorithms"
                ]
            },
            {
                "action": "Debug sensor configuration",
                "priority": "HIGH", 
                "time_estimate": "1 hour",
                "steps": [
                    "Check AirSim settings for depth camera",
                    "Verify sensor update rates and thresholds",
                    "Test manual obstacle detection",
                    "Validate sensor data pipeline"
                ]
            },
            {
                "action": "Add algorithm debugging",
                "priority": "HIGH",
                "time_estimate": "2 hours", 
                "steps": [
                    "Add debug logging to segmentation algorithms",
                    "Print intermediate processing steps",
                    "Test with synthetic obstacle data",
                    "Validate gap detection logic"
                ]
            }
        ],
        "success_criteria": [
            "Gap detection rate > 50%",
            "Successful obstacle detection in test environment",
            "Meaningful gap measurements (not all NaN)",
            "Processing rate > 10 FPS"
        ],
        "timeline": "Fix within 24-48 hours for system to be functional"
    }
    
    # Save action plan
    with open("logs/urgent_action_plan.json", 'w') as f:
        json.dump(action_plan, f, indent=2)
    
    print("✓ Created urgent action plan: logs/urgent_action_plan.json")

def main():
    diagnose_segmentation_issue()
    print("\n" + "=" * 45)
    create_debug_segmentation_script()
    print("\n" + "=" * 45)
    create_immediate_action_plan()
    
    print("\n" + "=" * 45)
    print("SUMMARY")
    print("=" * 45)
    print("Your segmentation system has a critical issue:")
    print("• 0% gap detection rate across all algorithms")
    print("• All obstacle sensors reading 1.0 (no obstacles)")
    print("• Gap detection requires obstacles to work")
    print()
    print("NEXT IMMEDIATE STEPS:")
    print("1. Run debug_segmentation.py")
    print("2. Test in environment with obstacles")
    print("3. Fix sensor configuration if needed")
    print("4. Re-run segmentation analysis")
    print()
    print("Expected time to fix: 2-4 hours")
    print("This should be your #1 priority!")

if __name__ == "__main__":
    main()