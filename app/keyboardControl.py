import tkinter as tk
from services import SimulationService
from UI import RobotApp

def main():
    # 1. Create main window
    root = tk.Tk()
    
    # 2. Instantiate state services (handles map & agent APIs)
    service = SimulationService(map_size= 18)
    
    # 3. Instantiate UI, injecting the service layer dependency
    app = RobotApp(root, service)
    
    # 4. Start the Tkinter application loop
    root.mainloop()

if __name__ == "__main__":
    main()
