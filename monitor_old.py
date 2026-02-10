import asyncio
import sys
import shutil
from mavsdk import System
from mavsdk.telemetry import FlightMode

def print_dashboard(pos, vel, att, batt, mode, lidar, gps):
    # Clear screen
    print("\033[H\033[J", end="")
    
    width = 60
    print("=" * width)
    print(f"   LESNAR AI - DRONE MONITOR (WSL/Gazebo)   ")
    print("=" * width)
    
    print(f" Status:      {'CONNECTED' if pos else 'WAITING...'}")
    print(f" Flight Mode: {mode}")
    print(f" Battery:     {batt:.1f} V")
    print("-" * width)
    
    if pos:
        print(f" Position:")
        print(f"   Lat: {pos.latitude_deg:.6f}")
        print(f"   Lon: {pos.longitude_deg:.6f}")
        print(f"   Alt: {pos.relative_altitude_m:.1f} m (Rel)")
    
    if vel:
        speed = (vel.north_m_s**2 + vel.east_m_s**2)**0.5
        print(f" Velocity:")
        print(f"   North: {vel.north_m_s:.1f} m/s")
        print(f"   East:  {vel.east_m_s:.1f} m/s")
        print(f"   Down:  {vel.down_m_s:.1f} m/s")
        print(f"   Speed: {speed:.1f} m/s")

    print("-" * width)
    if lidar:
        min_dist = min([x for x in lidar.distances if x > 0], default=999)
        print(f" LIDAR 360 (Blind Bat Eye):")
        print(f"   Min Dist: {min_dist:.2f} m")
        # Visualize segments
        # ...
    else:
        print(" LIDAR: No Data (Waiting for sensor...)")

    print("-" * width)
    print(" Press Ctrl+C to exit")

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    asyncio.ensure_future(print_loop(drone))
    
    # Keep alive
    while True:
        await asyncio.sleep(1)

async def print_loop(drone):
    pos = None
    vel = None
    att = None
    batt = 0.0
    mode = "UNKNOWN"
    lidar = None
    gps = None

    # We need to run these concurrently or poll them
    # For simplicity in this display script, we'll just subscribe to everything and update vars
    # A cleaner way is using multiple tasks, but let's try a simple polling loop for dashboard
    
    # Start background tasks
    asyncio.create_task(update_pos(drone, lambda x: exec("nonlocal pos; pos = x")))
    asyncio.create_task(update_vel(drone, lambda x: exec("nonlocal vel; vel = x")))
    asyncio.create_task(update_batt(drone, lambda x: exec("nonlocal batt; batt = x")))
    asyncio.create_task(update_mode(drone, lambda x: exec("nonlocal mode; mode = x")))
    asyncio.create_task(update_lidar(drone, lambda x: exec("nonlocal lidar; lidar = x")))

    while True:
        # Update vars from tasks (scope hack via closure/nonlocal isn't ideal in py, 
        # but we use mutable dict for state usually)
        # Let's just use a state dict
        print_dashboard(STATE['pos'], STATE['vel'], STATE['att'], STATE['batt'], STATE['mode'], STATE['lidar'], STATE['gps'])
        await asyncio.sleep(0.5)

STATE = {
    'pos': None, 'vel': None, 'att': None, 'batt': 0.0, 'mode': "UNKNOWN", 'lidar': None, 'gps': None
}

async def update_pos(drone):
    async for x in drone.telemetry.position(): STATE['pos'] = x
async def update_vel(drone):
    async for x in drone.telemetry.velocity_ned(): STATE['vel'] = x
async def update_batt(drone):
    async for x in drone.telemetry.battery(): STATE['batt'] = x.voltage_v
async def update_mode(drone):
    async for x in drone.telemetry.flight_mode(): STATE['mode'] = str(x)
async def update_lidar(drone):
    async for x in drone.telemetry.obstacle_distance(): STATE['lidar'] = x
    
async def update_tasks():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    t1 = asyncio.create_task(update_pos(drone))
    t2 = asyncio.create_task(update_vel(drone))
    t3 = asyncio.create_task(update_batt(drone))
    t4 = asyncio.create_task(update_mode(drone))
    t5 = asyncio.create_task(update_lidar(drone))
    
    while True:
        print_dashboard(STATE['pos'], STATE['vel'], STATE['att'], STATE['batt'], STATE['mode'], STATE['lidar'], STATE['gps'])
        await asyncio.sleep(0.2)
        
if __name__ == "__main__":
    try:
        asyncio.run(update_tasks())
    except KeyboardInterrupt:
        pass
    t3 = asyncio.create_task(update_batt(drone))
    t4 = asyncio.create_task(update_mode(drone))
    t5 = asyncio.create_task(update_lidar(drone))
    
    while True:
        print_dashboard(STATE['pos'], STATE['vel'], STATE['att'], STATE['batt'], STATE['mode'], STATE['lidar'], STATE['gps'])
        await asyncio.sleep(0.2)
