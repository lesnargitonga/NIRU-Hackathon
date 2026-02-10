import asyncio
from mavsdk import System
import sys

print("Imported System", flush=True)

async def run():
    print("Creating Drone (default)", flush=True)
    drone = System()
    print("Connecting to udp://:14540", flush=True)
    await drone.connect(system_address="udp://:14540")
    print("Connected", flush=True)

    print("Checking state", flush=True)
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"State connected: {state.is_connected}", flush=True)
            break

print("Starting Loop", flush=True)
if __name__ == "__main__":
    asyncio.run(run())
