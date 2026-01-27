import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

print(f"Python path: {sys.path[0]}")

try:
    import cv2
    print("cv2 imported successfully")
except ImportError as e:
    print(f"cv2 import FAILED: {e}")

try:
    import numpy as np
    print("numpy imported successfully")
except ImportError as e:
    print(f"numpy import FAILED: {e}")

try:
    from ai_modules.computer_vision import ComputerVisionSystem
    print("ai_modules.computer_vision imported successfully")
    cvsys = ComputerVisionSystem()
    print("ComputerVisionSystem initialized successfully")
except Exception as e:
    print(f"CV system load FAILED: {e}")
    import traceback
    traceback.print_exc()
