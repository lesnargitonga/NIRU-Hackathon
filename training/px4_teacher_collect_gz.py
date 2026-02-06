import asyncio
import argparse
import csv
import json
import math
import time
import heapq
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    # Only needed for online / PX4 mode
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityNedYaw
except Exception:
    System = None
    OffboardError = Exception
    VelocityNedYaw = None

"""
GOD-MODE TEACHER COLLECTOR
--------------------------
1. Parses obstacles.sdf to load the EXACT world map.
2. Uses this 'ground truth' to simulate a PERFECT LiDAR scan.
3. Why? Because the real Gazebo Lidar has been unreliable (stuck at 20m).
4. Saves this clean, noise-free simulated Lidar data for the student to learn from.
"""

# --- UTILS ---
def wrap_deg(a: float) -> float:
    return (a + 180) % 360 - 180

def shortest_diff(current: float, target: float) -> float:
    return wrap_deg(target - current)

class Obstacle:
    def __init__(self, x, y, radius, height, is_box=False, dx=0, dy=0):
        self.x = x
        self.y = y
        self.radius = radius # For cylinder
        self.height = height
        self.is_box = is_box
        self.dx = dx # For box
        self.dy = dy # For box

    def distance_to_point(self, px, py):
        if not self.is_box:
            dist = math.sqrt((px - self.x)**2 + (py - self.y)**2) - self.radius
            return max(0.0, dist)
        else:
            tx = abs(px - self.x)
            ty = abs(py - self.y)
            dx = max(tx - (self.dx/2), 0)
            dy = max(ty - (self.dy/2), 0)
            return math.sqrt(dx*dx + dy*dy)

    def is_inside(self, px, py, margin=0.0):
        if not self.is_box:
            # Cylinder check
            dist_sq = (px - self.x)**2 + (py - self.y)**2
            return dist_sq < (self.radius + margin)**2
        else:
            # Box check
            # Axis aligned assumption for simplicity in teacher
            half_x = (self.dx / 2) + margin
            half_y = (self.dy / 2) + margin
            return (abs(px - self.x) < half_x) and (abs(py - self.y) < half_y)


class Map:
    def __init__(self, sdf_path):
        self.obstacles = []
        self.load_sdf(sdf_path)

    def load_sdf(self, path):
        print(f"Loading map from {path}...")
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            world = root.find("world")
            for model in world.findall("model"):
                name = model.get("name")
                # Look for pose
                pose_elem = model.find("pose")
                if pose_elem is None: continue
                pose_str = pose_elem.text
                parts = [float(f) for f in pose_str.split()]
                mx, my, mz = parts[0], parts[1], parts[2]
                
                link = model.find("link")
                if link is None: continue
                collision = link.find("collision")
                if collision is None: continue
                geometry = collision.find("geometry")
                if geometry is None: continue
                
                if geometry.find("box") is not None:
                    size_str = geometry.find("box").find("size").text
                    dims = [float(f) for f in size_str.split()]
                    # Box obstacle
                    self.obstacles.append(Obstacle(mx, my, 0, dims[2], True, dims[0], dims[1]))
                    
                elif geometry.find("cylinder") is not None:
                    cyl = geometry.find("cylinder")
                    r = float(cyl.find("radius").text)
                    h = float(cyl.find("length").text)
                    self.obstacles.append(Obstacle(mx, my, r, h, False))
                    
            print(f"Loaded {len(self.obstacles)} obstacles.")
            
        except Exception as e:
            print(f"Failed to load map: {e}")

    def simulate_lidar(self, px, py, pz, yaw_deg, num_rays=72, max_dist=20.0):
        # Raycast against all obstacles
        # To be fast, we just calculate distance to center - size
        # This is a 'closest point' lidar, slightly different from raycast
        # but for cylinders/boxes it is close enough for training avoidance
    async def collect_data(args):
        """ONLINE MODE: PX4 + Gazebo + MAVSDK.

        Requires WSL / PX4 / Gazebo to be running. If you don't want that,
        run with --offline true and it will stay entirely in Python.
        """

        if System is None:
            raise RuntimeError("MAVSDK not available in this environment. Use --offline true.")

        # 1. Map & Grid
        world_map = Map(args.sdf_path)
        grid = GridMap(world_map.obstacles, resolution=1.0, margin=1.5)  # 1m res, 1.5m margin

        # Connect
        drone = System()
        addr = args.system
        if "://" not in addr:
            addr = f"udpin://{addr}"
        print(f"--> Connecting to {addr}...")
        await drone.connect(system_address=addr)

        print("--> Waiting for drone...")
        async for s in drone.core.connection_state():
            if s.is_connected:
                break
        print("--> Connected!")

        dstate = DroneState()
        asyncio.create_task(telemetry_listener(drone, dstate))
        asyncio.create_task(velocity_listener(drone, dstate))
        asyncio.create_task(att_listener(drone, dstate))

        await asyncio.sleep(2)
        print("--> Arming...")
        try:
            await drone.action.arm()
            await drone.action.set_takeoff_altitude(args.alt)
            await drone.action.takeoff()
            await asyncio.sleep(8)
        except Exception as e:
            print(f"Arm/Takeoff failed: {e}")

        # Wait for valid GPS
        while dstate.lat == 0:
            await asyncio.sleep(0.1)
        home_lat = dstate.lat
        home_lon = dstate.lon
        print(f"--> Home set at {home_lat:.6f}, {home_lon:.6f}")

        def get_local_pos():
            dlat = dstate.lat - home_lat
            dlon = dstate.lon - home_lon
            y = dlon * 111319.0 * math.cos(math.radians(home_lat))
            x = dlat * 111132.0
            return x, y

        # Start Offboard
        print("--> Starting A* Pathfinding (ONLINE)...")
        try:
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, dstate.yaw)
            )
            await drone.offboard.start()
        except OffboardError as e:
            print(e)
            return

        # CSV Logging
        f = open(args.out, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        header = [
            "timestamp",
            "lat",
            "lon",
            "rel_alt",
            "vx",
            "vy",
            "vz",
            "yaw",
            "cmd_vx",
            "cmd_vy",
            "cmd_vz",
            "cmd_yaw",
            "lidar_min",
            "lidar_json",
            "goal_x",
            "goal_y",
        ]
        writer.writerow(header)

        start_time = time.time()
        last_step = time.time()

        current_path = []
        path_index = 0
        goal_x, goal_y = 0, 0

        def pick_new_goal():
            while True:
                gx = np.random.randint(10, grid.width - 10)
                gy = np.random.randint(10, grid.height - 10)
                if not grid.is_blocked(gx, gy):
                    wx, wy = grid.grid_to_world(gx, gy)
                    curr_x, curr_y = get_local_pos()
                    dist = math.sqrt((wx - curr_x) ** 2 + (wy - curr_y) ** 2)
                    if dist > 30:
                        return wx, wy

        goal_x, goal_y = pick_new_goal()
        print(f"First Goal: {goal_x:.1f}, {goal_y:.1f}")

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > args.duration:
                    break

                now = time.time()
                dt = now - last_step
                if dt < 0.1:
                    await asyncio.sleep(0.01)
                    continue
                last_step = now

                px, py = get_local_pos()

                if not current_path or path_index >= len(current_path):
                    print(f"Planning A* to {goal_x:.0f},{goal_y:.0f}...")
                    path = astar(grid, (px, py), (goal_x, goal_y))
                    if not path:
                        print("Path failed! Picking new goal.")
                        goal_x, goal_y = pick_new_goal()
                        continue

                    current_path = path[::2] + [path[-1]]
                    path_index = 0
                    print(f"Path found! {len(current_path)} waypoints")

                lookahead_dist = 4.0
                target_pt = current_path[path_index]

                for i in range(path_index, len(current_path)):
                    pt = current_path[i]
                    d = math.sqrt((pt[0] - px) ** 2 + (pt[1] - py) ** 2)
                    if d > lookahead_dist:
                        target_pt = pt
                        path_index = i
                        break

                    if i == len(current_path) - 1 and d < 2.0:
                        print("Goal Reached! New Goal...")
                        goal_x, goal_y = pick_new_goal()
                        current_path = []
                        break

                if not current_path:
                    continue

                tx, ty = target_pt
                dx = tx - px
                dy = ty - py
                dist = math.sqrt(dx * dx + dy * dy)

                desired_speed = args.max_speed
                vx_cmd = (dx / dist) * desired_speed if dist > 0 else 0.0
                vy_cmd = (dy / dist) * desired_speed if dist > 0 else 0.0

                target_yaw = math.degrees(math.atan2(dy, dx))
                diff = shortest_diff(dstate.yaw, target_yaw)

                sim_lidar = world_map.simulate_lidar(px, py, dstate.rel_alt, dstate.yaw)
                min_dist = float(np.min(sim_lidar))

                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(vx_cmd, vy_cmd, 0.0, dstate.yaw + diff)
                )

                row = [
                    time.time(),
                    dstate.lat,
                    dstate.lon,
                    dstate.rel_alt,
                    dstate.vx,
                    dstate.vy,
                    dstate.vz,
                    dstate.yaw,
                    vx_cmd,
                    vy_cmd,
                    0.0,
                    0.0,
                    min_dist,
                    json.dumps(sim_lidar.tolist()),
                    goal_x,
                    goal_y,
                ]
                writer.writerow(row)
        finally:
            try:
                await drone.action.land()
            except Exception:
                pass
            f.close()
            print("Done (ONLINE mode)!")


    def collect_data_offline(args):
        """OFFLINE MODE: Pure Python expert trajectories.

        - No PX4
        - No Gazebo
        - No WSL

        We simulate a point-mass drone moving in the 2D map using
        the same A* + pure-pursuit logic, and we still generate
        lidar + action pairs for training.
        """

        world_map = Map(args.sdf_path)
        grid = GridMap(world_map.obstacles, resolution=1.0, margin=1.5)

        f = open(args.out, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        header = [
            "timestamp",
            "lat",
            "lon",
            "rel_alt",
            "vx",
            "vy",
            "vz",
            "yaw",
            "cmd_vx",
            "cmd_vy",
            "cmd_vz",
            "cmd_yaw",
            "lidar_min",
            "lidar_json",
            "goal_x",
            "goal_y",
        ]
        writer.writerow(header)

        # Fake state in local frame
        px, py = 0.0, 0.0
        yaw = 0.0
        alt = args.alt
        vx = vy = vz = 0.0

        start_time = time.time()
        last_step = time.time()

        current_path = []
        path_index = 0
        goal_x, goal_y = 0.0, 0.0

        def pick_new_goal_offline(cx, cy):
            while True:
                gx = np.random.randint(10, grid.width - 10)
                gy = np.random.randint(10, grid.height - 10)
                if not grid.is_blocked(gx, gy):
                    wx, wy = grid.grid_to_world(gx, gy)
                    dist = math.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
                    if dist > 30:
                        return wx, wy

        goal_x, goal_y = pick_new_goal_offline(px, py)
        print(f"[OFFLINE] First Goal: {goal_x:.1f}, {goal_y:.1f}")

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > args.duration:
                    break

                now = time.time()
                dt = now - last_step
                if dt < 0.1:
                    time.sleep(0.01)
                    continue
                last_step = now

                # 1. Re-plan if needed
                if not current_path or path_index >= len(current_path):
                    print(f"[OFFLINE] Planning A* to {goal_x:.0f},{goal_y:.0f}...")
                    path = astar(grid, (px, py), (goal_x, goal_y))
                    if not path:
                        print("[OFFLINE] Path failed! Picking new goal.")
                        goal_x, goal_y = pick_new_goal_offline(px, py)
                        continue

                    current_path = path[::2] + [path[-1]]
                    path_index = 0
                    print(f"[OFFLINE] Path found! {len(current_path)} waypoints")

                # 2. Pure-pursuit along path
                lookahead_dist = 4.0
                target_pt = current_path[path_index]

                for i in range(path_index, len(current_path)):
                    pt = current_path[i]
                    d = math.sqrt((pt[0] - px) ** 2 + (pt[1] - py) ** 2)
                    if d > lookahead_dist:
                        target_pt = pt
                        path_index = i
                        break

                    if i == len(current_path) - 1 and d < 2.0:
                        print("[OFFLINE] Goal Reached! New Goal...")
                        goal_x, goal_y = pick_new_goal_offline(px, py)
                        current_path = []
                        break

                if not current_path:
                    continue

                tx, ty = target_pt
                dx = tx - px
                dy = ty - py
                dist = math.sqrt(dx * dx + dy * dy)

                desired_speed = args.max_speed
                cmd_vx = (dx / dist) * desired_speed if dist > 0 else 0.0
                cmd_vy = (dy / dist) * desired_speed if dist > 0 else 0.0

                target_yaw = math.degrees(math.atan2(dy, dx))
                yaw_err = shortest_diff(yaw, target_yaw)
                cmd_yaw_rate = max(-45.0, min(45.0, yaw_err * 2.0))

                # 3. Integrate simple kinematics
                yaw = wrap_deg(yaw + cmd_yaw_rate * dt)
                px += cmd_vx * dt
                py += cmd_vy * dt
                vx = cmd_vx
                vy = cmd_vy
                vz = 0.0

                # 4. Simulated lidar
                sim_lidar = world_map.simulate_lidar(px, py, alt, yaw)
                min_dist = float(np.min(sim_lidar))

                row = [
                    time.time(),
                    0.0,  # fake lat
                    0.0,  # fake lon
                    alt,
                    vx,
                    vy,
                    vz,
                    yaw,
                    cmd_vx,
                    cmd_vy,
                    0.0,
                    cmd_yaw_rate,
                    min_dist,
                    json.dumps(sim_lidar.tolist()),
                    goal_x,
                    goal_y,
                ]
                writer.writerow(row)
        finally:
            f.close()
            print("Done (OFFLINE mode)!")
class DroneState:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.rel_alt = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 0.0
        self.lidar_ranges = []
        self.lidar_min = 20.0

# --- LISTENERS ---
async def telemetry_listener(drone, state):
    async for pos in drone.telemetry.position():
        state.lat = pos.latitude_deg
        state.lon = pos.longitude_deg
        state.rel_alt = pos.relative_altitude_m

async def velocity_listener(drone, state):
    async for vel in drone.telemetry.velocity_ned():
        state.vx = vel.north_m_s
        state.vy = vel.east_m_s
        state.vz = vel.down_m_s

async def att_listener(drone, state):
    async for angle in drone.telemetry.heading():
        state.yaw = angle.heading_deg


# --- MAIN ---
async def collect_data(args):
    # 1. Map & Grid
    world_map = Map(args.sdf_path)
    grid = GridMap(world_map.obstacles, resolution=1.0, margin=1.5) # 1m res, 1.5m margin
    
    # Connect
    drone = System()
    addr = args.system
    if "://" not in addr: addr = f"udpin://{addr}"
    print(f"--> Connecting to {addr}...")
    await drone.connect(system_address=addr)
    
    print("--> Waiting for drone...")
    async for s in drone.core.connection_state():
        if s.is_connected: break
    print("--> Connected!")

    dstate = DroneState()
    asyncio.create_task(telemetry_listener(drone, dstate))
    asyncio.create_task(velocity_listener(drone, dstate))
    asyncio.create_task(att_listener(drone, dstate))
    
    await asyncio.sleep(2)
    print("--> Arming...")
    try:
        await drone.action.arm()
        await drone.action.set_takeoff_altitude(args.alt)
        await drone.action.takeoff()
        await asyncio.sleep(8)
    except: pass

    while dstate.lat == 0: await asyncio.sleep(0.1)
    home_lat = dstate.lat
    home_lon = dstate.lon
    print(f"--> Home set at {home_lat:.6f}, {home_lon:.6f}")
    
    def get_local_pos():
        dlat = dstate.lat - home_lat
        dlon = dstate.lon - home_lon
        y = dlon * 111319.0 * math.cos(math.radians(home_lat))
        x = dlat * 111132.0 
        return x, y

    # Start Offboard
    print("--> Starting A* Pathfinding...")
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, dstate.yaw))
        await drone.offboard.start()
    except OffboardError as e:
        print(e)
        return

    # CSV Logging
    f = open(args.out, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    header = ["timestamp", "lat", "lon", "rel_alt", "vx", "vy", "vz", "yaw",
              "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yaw", "lidar_min", "lidar_json", "goal_x", "goal_y"]
    writer.writerow(header)

    start_time = time.time()
    last_step = time.time()
    
    current_path = []
    path_index = 0
    goal_x, goal_y = 0, 0

    def pick_new_goal():
        while True:
            # Pick goal within grid bounds
            # Random exploration
            gx = np.random.randint(10, grid.width-10)
            gy = np.random.randint(10, grid.height-10)
            if not grid.is_blocked(gx, gy):
                wx, wy = grid.grid_to_world(gx, gy)
                # Ensure it's far enough
                curr_x, curr_y = get_local_pos()
                dist = math.sqrt((wx-curr_x)**2 + (wy-curr_y)**2)
                if dist > 30: # Long missions
                    return wx, wy

    # Initial Goal
    goal_x, goal_y = pick_new_goal()
    print(f"First Goal: {goal_x:.1f}, {goal_y:.1f}")

    while True:
        elapsed = time.time() - start_time
        if elapsed > args.duration: break
        
        # 1. Loop Timing
        now = time.time()
        dt = now - last_step
        if dt < 0.1:
            await asyncio.sleep(0.01)
            continue
        last_step = now
        
        px, py = get_local_pos()
        
        # 2. Path Planning Check
        if not current_path or path_index >= len(current_path):
            print(f"Planning A* to {goal_x:.0f},{goal_y:.0f}...")
            # Plan from current pos to goal
            path = astar(grid, (px, py), (goal_x, goal_y))
            if not path:
                print("Path failed! Picking new goal.")
                goal_x, goal_y = pick_new_goal()
                continue
            
            # Subsample path (Teacher can smooth it)
            # A* returns all grid cells, we can skip some
            current_path = path[::2] + [path[-1]]
            path_index = 0
            print(f"Path found! {len(current_path)} waypoints")

        # 3. Path Following (Pure Pursuit)
        # Look ahead
        lookahead_dist = 4.0
        target_pt = current_path[path_index]
        
        # Find point on path ahead of us
        for i in range(path_index, len(current_path)):
            pt = current_path[i]
            d = math.sqrt((pt[0]-px)**2 + (pt[1]-py)**2)
            if d > lookahead_dist:
                target_pt = pt
                path_index = i # Advance index
                break
            
            # If we are close to end
            if i == len(current_path) - 1 and d < 2.0:
                # Goal reached
                print("Goal Reached! New Goal...")
                goal_x, goal_y = pick_new_goal()
                current_path = []
                break
        
        if not current_path: continue # Re-plan next loop
        
        # Drive to target_pt
        tx, ty = target_pt
        dx = tx - px
        dy = ty - py
        dist = math.sqrt(dx*dx + dy*dy)
        
        desired_speed = args.max_speed
        
        # Simple P-Control for Velocity
        # Normalize direction
        vx_cmd = (dx / dist) * desired_speed if dist > 0 else 0
        vy_cmd = (dy / dist) * desired_speed if dist > 0 else 0
        
        # Yaw slightly towards target
        target_yaw = math.degrees(math.atan2(dy, dx))
        diff = shortest_diff(dstate.yaw, target_yaw)
        
        # 4. Simulate Sensors (For the Dataset)
        sim_lidar = world_map.simulate_lidar(px, py, dstate.rel_alt, dstate.yaw)
        min_dist = np.min(sim_lidar)
        
        # 5. Command
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vx_cmd, vy_cmd, 0.0, dstate.yaw + diff))
        
        # 6. Log
        row = [
            time.time(), dstate.lat, dstate.lon, dstate.rel_alt,
            dstate.vx, dstate.vy, dstate.vz, dstate.yaw,
            vx_cmd, vy_cmd, 0.0, 0.0, min_dist, json.dumps(sim_lidar.tolist()), goal_x, goal_y
        ]

        writer.writerow(row)
        
    await drone.action.land()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="dataset/px4_teacher/telemetry_god.csv")
    parser.add_argument("--system", type=str, default="udpin://0.0.0.0:14540")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--alt", type=float, default=15.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--base_speed", type=float, default=3.0)
    parser.add_argument("--max_speed", type=float, default=8.0)
    parser.add_argument("--yaw_rate_limit", type=float, default=40.0)
    parser.add_argument("--sdf_path", type=str, default=r"d:\docs\lesnar\Lesnar AI\obstacles.sdf")
    parser.add_argument("--offline", action="store_true", help="Run without PX4/Gazebo (pure Python)")
    
    args = parser.parse_args()
    if args.offline:
        collect_data_offline(args)
    else:
        asyncio.run(collect_data(args))

