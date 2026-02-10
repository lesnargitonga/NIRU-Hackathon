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
from datetime import datetime, timezone

try:
    # Only needed for online / PX4 mode
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityNedYaw
except Exception:
    System = None
    OffboardError = Exception
    VelocityNedYaw = None

LOG_PATH = "teacher_runtime.log"

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

log("Teacher script started")

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
        self.radius = radius  # For cylinder
        self.height = height
        self.is_box = is_box
        self.dx = dx  # For box
        self.dy = dy  # For box

    def distance_to_point(self, px, py):
        if not self.is_box:
            dist = math.sqrt((px - self.x) ** 2 + (py - self.y) ** 2) - self.radius
            return max(0.0, dist)
        else:
            tx = abs(px - self.x)
            ty = abs(py - self.y)
            dx = max(tx - (self.dx / 2), 0)
            dy = max(ty - (self.dy / 2), 0)
            return math.sqrt(dx * dx + dy * dy)

    def is_inside(self, px, py, margin=0.0):
        if not self.is_box:
            dist_sq = (px - self.x) ** 2 + (py - self.y) ** 2
            return dist_sq < (self.radius + margin) ** 2
        else:
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
                pose_elem = model.find("pose")
                if pose_elem is None:
                    continue
                parts = [float(f) for f in pose_elem.text.split()]
                mx, my, mz = parts[0], parts[1], parts[2]

                link = model.find("link")
                if link is None:
                    continue
                collision = link.find("collision")
                if collision is None:
                    continue
                geometry = collision.find("geometry")
                if geometry is None:
                    continue

                if geometry.find("box") is not None:
                    size_str = geometry.find("box").find("size").text
                    dims = [float(f) for f in size_str.split()]
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
        ranges = np.ones(num_rays) * max_dist
        fov = 360.0
        angle_step = fov / num_rays

        nearby = []
        for obs in self.obstacles:
            dist_center = math.sqrt((px - obs.x) ** 2 + (py - obs.y) ** 2)
            max_dim = max(obs.dx, obs.dy) if obs.is_box else obs.radius * 2
            if dist_center - (max_dim / 2) < max_dist:
                nearby.append(obs)

        yaw_rad = math.radians(yaw_deg)

        for i in range(num_rays):
            rel_angle = -180 + (i * angle_step)
            ray_angle = math.radians(rel_angle) + yaw_rad
            rx = math.cos(ray_angle)
            ry = math.sin(ray_angle)

            min_hit = max_dist

            for obs in nearby:
                ox = obs.x - px
                oy = obs.y - py

                dot = ox * rx + oy * ry
                if dot > 0:
                    cross = abs(ox * ry - oy * rx)
                    size = (max(obs.dx, obs.dy) / 2) if obs.is_box else obs.radius
                    if cross < size:
                        dist = dot - size
                        if dist < min_hit:
                            min_hit = dist

            ranges[i] = max(0.0, min_hit)

        return ranges


# --- PATHFINDING ---
class GridMap:
    def __init__(self, obstacles, resolution=1.0, margin=1.5):
        self.res = resolution
        pad = 20
        all_x = [o.x for o in obstacles] + [0]
        all_y = [o.y for o in obstacles] + [0]
        self.min_x = min(all_x) - pad
        self.max_x = max(all_x) + pad
        self.min_y = min(all_y) - pad
        self.max_y = max(all_y) + pad

        self.width = int((self.max_x - self.min_x) / self.res)
        self.height = int((self.max_y - self.min_y) / self.res)

        print(
            f"Grid Size: {self.width}x{self.height}, Bounds: x[{self.min_x:.1f},{self.max_x:.1f}] y[{self.min_y:.1f},{self.max_y:.1f}]"
        )

        self.grid = np.zeros((self.width, self.height), dtype=bool)

        print("Rasterizing obstacles...")
        for o in obstacles:
            size = (max(o.dx, o.dy) if o.is_box else o.radius * 2) + (margin * 2)
            steps = int(size / self.res) + 2

            cx, cy = self.world_to_grid(o.x, o.y)

            for i in range(-steps, steps + 1):
                for j in range(-steps, steps + 1):
                    gx, gy = cx + i, cy + j
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        wx, wy = self.grid_to_world(gx, gy)
                        if o.is_inside(wx, wy, margin):
                            self.grid[gx, gy] = True

    def world_to_grid(self, x, y):
        gx = int((x - self.min_x) / self.res)
        gy = int((y - self.min_y) / self.res)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = (gx * self.res) + self.min_x
        wy = (gy * self.res) + self.min_y
        return wx, wy

    def is_blocked(self, gx, gy):
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return True
        return self.grid[gx, gy]


def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def astar(grid_map, start, goal):
    start_node = grid_map.world_to_grid(start[0], start[1])
    goal_node = grid_map.world_to_grid(goal[0], goal[1])

    if grid_map.is_blocked(*goal_node):
        print("WARN: Goal is blocked!")
        return []

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    g_score = {start_node: 0}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal_node:
            path = []
            while current in came_from:
                path.append(grid_map.grid_to_world(*current))
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if grid_map.is_blocked(*neighbor):
                continue

            tentative_g = g_score[current] + math.sqrt(dx * dx + dy * dy)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f, neighbor))

    return []


# --- DRONE STATE ---
class DroneState:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.rel_alt = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 0.0


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


async def local_pos_listener(drone, state):
    async for pv in drone.telemetry.position_velocity_ned():
        state.x = pv.position.north_m
        state.y = pv.position.east_m
        state.z = pv.position.down_m
        state.vx = pv.velocity.north_m_s
        state.vy = pv.velocity.east_m_s
        state.vz = pv.velocity.down_m_s


async def att_listener(drone, state):
    async for angle in drone.telemetry.heading():
        state.yaw = angle.heading_deg


# --- MAIN (ONLINE) ---
async def collect_data(args):
    """ONLINE MODE: PX4 + Gazebo + MAVSDK."""

    if System is None:
        raise RuntimeError("MAVSDK not available in this environment. Use --offline.")

    log("--> Starting teacher (ONLINE)...")
    world_map = Map(args.sdf_path)
    log("--> Map loaded.")
    grid = GridMap(world_map.obstacles, resolution=1.0, margin=1.5)
    log("--> Grid ready.")

    drone = System(
        mavsdk_server_address=args.mavsdk_server,
        port=args.mavsdk_port,
    )
    addr = args.system
    if "://" not in addr:
        addr = f"udpin://{addr}"
    log(f"--> Connecting to {addr}...")
    try:
        await asyncio.wait_for(drone.connect(system_address=addr), timeout=10)
    except asyncio.TimeoutError:
        raise RuntimeError("MAVSDK connect() timed out. Is PX4 running and reachable?")
    except Exception as e:
        log(f"!! connect() failed: {e}")
        raise

    log("--> Waiting for drone (timeout 30s)...")
    start_wait = time.time()
    try:
        async for s in drone.core.connection_state():
            if s.is_connected:
                log("--> Connected!")
                break
            if time.time() - start_wait > 30:
                raise RuntimeError(
                    "Timed out waiting for PX4. Check MAVLink UDP reachability and PX4 is running."
                )
    except Exception as e:
        log(f"!! connection_state failed: {e}")
        raise

    dstate = DroneState()
    asyncio.create_task(telemetry_listener(drone, dstate))
    asyncio.create_task(local_pos_listener(drone, dstate))
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

    # Prefer local NED position (stable in SITL) over GPS-derived xy.
    # Wait briefly until local position starts updating.
    start_lp = time.time()
    while time.time() - start_lp < 10.0 and (dstate.x == 0.0 and dstate.y == 0.0):
        await asyncio.sleep(0.05)

    def get_local_pos():
        return dstate.x, dstate.y

    print("--> Starting A* Pathfinding (ONLINE)...")
    try:
        # PX4 offboard best practice: stream setpoints before starting offboard.
        warmup_hz = max(5.0, float(args.hz))
        warmup_count = int(warmup_hz * 1.0)
        for _ in range(warmup_count):
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, dstate.yaw))
            await asyncio.sleep(1.0 / warmup_hz)
        await drone.offboard.start()
    except OffboardError as e:
        print(e)
        return

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

    # Command smoothing / limits (stability)
    prev_vx_cmd = 0.0
    prev_vy_cmd = 0.0
    max_accel = 2.0  # m/s^2

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
            period = 1.0 / max(1.0, float(args.hz))
            if dt < period:
                await asyncio.sleep(max(0.0, period - dt))
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

            # Desired speed tapers down near the target to reduce oscillation.
            desired_speed = min(float(args.max_speed), max(float(args.base_speed), 0.35 * dist))
            vx_des = (dx / dist) * desired_speed if dist > 1e-6 else 0.0
            vy_des = (dy / dist) * desired_speed if dist > 1e-6 else 0.0

            # Acceleration-limit + light low-pass to avoid twitch.
            dv = max_accel * max(dt, 1e-3)
            vx_cmd = max(prev_vx_cmd - dv, min(prev_vx_cmd + dv, vx_des))
            vy_cmd = max(prev_vy_cmd - dv, min(prev_vy_cmd + dv, vy_des))
            alpha = 0.25
            vx_cmd = (1 - alpha) * prev_vx_cmd + alpha * vx_cmd
            vy_cmd = (1 - alpha) * prev_vy_cmd + alpha * vy_cmd
            prev_vx_cmd, prev_vy_cmd = vx_cmd, vy_cmd

            target_yaw = math.degrees(math.atan2(dy, dx))
            diff = shortest_diff(dstate.yaw, target_yaw)

            # Limit yaw change per cycle to prevent violent spins.
            max_yaw_step = float(args.yaw_rate_limit) * max(dt, 1e-3)
            diff = max(-max_yaw_step, min(max_yaw_step, diff))
            cmd_yaw = wrap_deg(dstate.yaw + diff)

            sim_lidar = world_map.simulate_lidar(px, py, dstate.rel_alt, dstate.yaw)
            min_dist = float(np.min(sim_lidar))

            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(vx_cmd, vy_cmd, 0.0, cmd_yaw)
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
            try:
                await drone.offboard.stop()
            except Exception:
                pass
            await drone.action.land()
        except Exception:
            pass
        f.close()
        print("Done (ONLINE mode)!")


# --- MAIN (OFFLINE) ---
def collect_data_offline(args):
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

            yaw = wrap_deg(yaw + cmd_yaw_rate * dt)
            px += cmd_vx * dt
            py += cmd_vy * dt
            vx = cmd_vx
            vy = cmd_vy
            vz = 0.0

            sim_lidar = world_map.simulate_lidar(px, py, alt, yaw)
            min_dist = float(np.min(sim_lidar))

            row = [
                time.time(),
                0.0,
                0.0,
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
    parser.add_argument("--mavsdk-server", type=str, default="127.0.0.1")
    parser.add_argument("--mavsdk-port", type=int, default=50051)

    args = parser.parse_args()
    if args.offline:
        collect_data_offline(args)
    else:
        asyncio.run(collect_data(args))
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

