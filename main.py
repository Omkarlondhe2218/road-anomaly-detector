import sys
import tkinter as tk
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.app import RoadAnomalyGUI


def main():
    print("=" * 60)
    print(" Launching YOLOv8m Road Anomaly Detection System GUI...")
    print("=" * 60)

    root = tk.Tk()
    app = RoadAnomalyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
