import argparse
import time
from pathlib import Path
import sys

import airsim


def wait_for_airsim(ip: str, port: int, timeout: float = 60.0, interval: float = 1.0) -> airsim.MultirotorClient | None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            client = airsim.MultirotorClient(ip=ip, port=port)
            client.confirmConnection()
            return client
        except Exception as e:
            last_err = e
            time.sleep(interval)
    print(f"[ERR] AirSim not reachable on {ip}:{port} within {timeout}s: {last_err}")
    return None


def simple_move(client: airsim.MultirotorClient, alt: float, speed: float, dur: float) -> None:
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync(timeout_sec=10).join()
    client.moveToZAsync(alt, 2).join()
    print('[OK] Takeoff + Z hold')

    # Body-frame forward jog
    print('[MOVE] Forward body-frame')
    client.moveByVelocityZBodyFrameAsync(speed, 0.0, float(alt), float(dur)).join()
    client.moveByVelocityZBodyFrameAsync(0.0, 0.0, float(alt), 0.4).join()
    print('[MOVE] Backward body-frame')
    client.moveByVelocityZBodyFrameAsync(-speed, 0.0, float(alt), float(dur)).join()
    client.moveByVelocityZBodyFrameAsync(0.0, 0.0, float(alt), 0.4).join()

    client.hoverAsync().join()
    print('[DONE] Hovering; landing...')
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=41451)
    ap.add_argument('--alt', type=float, default=-4.0)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--dur', type=float, default=1.2)
    ap.add_argument('--wait', type=float, default=90.0, help='Max seconds to wait for AirSim to come up')
    args = ap.parse_args()

    client = wait_for_airsim(args.ip, args.port, timeout=args.wait)
    if client is None:
        sys.exit(2)

    try:
        simple_move(client, args.alt, args.speed, args.dur)
    except Exception as e:
        print('[ERR] Move failed:', e)
        sys.exit(3)


if __name__ == '__main__':
    main()
