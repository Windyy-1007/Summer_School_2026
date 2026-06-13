import tkinter as tk

from app.UI import RobotApp
from app.services import SimulationService


def main():
    root = tk.Tk()
    service = SimulationService(map_size=18)
    RobotApp(root, service)
    root.mainloop()


if __name__ == "__main__":
    main()
