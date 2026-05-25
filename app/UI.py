import tkinter as tk

class RobotApp:
    def __init__(self, root, service):
        self.root = root
        self.service = service
        
        self.root.title("LineMap Robot Simulator")
        self.root.geometry("900x980")
        self.root.configure(bg="#1a1a1a")
        
        self.log_message = "Simulator started. Use A W S D to control."
        
        # Build GUI
        self.setup_ui()
        
        # Bind Keyboard Keys
        self.root.bind('<w>', lambda e: self.handle_action('W'))
        self.root.bind('<s>', lambda e: self.handle_action('S'))
        self.root.bind('<a>', lambda e: self.handle_action('A'))
        self.root.bind('<d>', lambda e: self.handle_action('D'))
        self.root.bind('<W>', lambda e: self.handle_action('W'))
        self.root.bind('<S>', lambda e: self.handle_action('S'))
        self.root.bind('<A>', lambda e: self.handle_action('A'))
        self.root.bind('<D>', lambda e: self.handle_action('D'))
        
        self.draw_map()

    def setup_ui(self):
        # Header Label
        header = tk.Label(
            self.root, 
            text="ROBOT NAVIGATION SIMULATOR", 
            font=("Helvetica", 16, "bold"), 
            bg="#1a1a1a", 
            fg="#ffffff", 
            pady=15
        )
        header.pack()

        # Canvas for Drawing Map
        self.canvas_size = 750
        self.canvas = tk.Canvas(
            self.root, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg="#262626", 
            highlightthickness=0
        )
        self.canvas.pack(pady=10)

        # Status and Control Frame
        info_frame = tk.Frame(self.root, bg="#1a1a1a", pady=10)
        info_frame.pack(fill=tk.X, padx=40)

        # Position and Direction Labels
        self.pos_label = tk.Label(
            info_frame, 
            text=f"Position: ({self.service.get_robot_x()}, {self.service.get_robot_y()})", 
            font=("Helvetica", 12, "bold"), 
            bg="#1a1a1a", 
            fg="#00adb5"
        )
        self.pos_label.pack(side=tk.LEFT, padx=10)

        self.dir_label = tk.Label(
            info_frame, 
            text=f"Heading: {self.service.get_robot_direction_name().upper()}", 
            font=("Helvetica", 12, "bold"), 
            bg="#1a1a1a", 
            fg="#00adb5"
        )
        self.dir_label.pack(side=tk.RIGHT, padx=10)

        # Console Log Box
        log_frame = tk.Frame(self.root, bg="#1a1a1a")
        log_frame.pack(fill=tk.X, padx=40, pady=5)
        
        self.log_label = tk.Label(
            log_frame, 
            text=self.log_message, 
            font=("Consolas", 10), 
            bg="#121212", 
            fg="#a8ffb2", 
            anchor="w", 
            justify="left",
            padx=10, 
            pady=8,
            relief=tk.SUNKEN,
            bd=1
        )
        self.log_label.pack(fill=tk.X)

        # Keyboard Guide
        guide = tk.Label(
            self.root, 
            text="Controls: W (Forward) | S (Backward) | A (Turn Left) | D (Turn Right)", 
            font=("Helvetica", 10, "italic"), 
            bg="#1a1a1a", 
            fg="#888888", 
            pady=10
        )
        guide.pack()

    def handle_action(self, key):
        if key == 'W':
            success = self.service.forward()
            rx, ry = self.service.get_robot_x(), self.service.get_robot_y()
            if success:
                self.log_message = f"Moved forward successfully to ({rx}, {ry})."
            else:
                self.log_message = "Move forward FAILED! Obstacle detected or out of bounds."
        elif key == 'S':
            success = self.service.backward()
            rx, ry = self.service.get_robot_x(), self.service.get_robot_y()
            if success:
                self.log_message = f"Moved backward successfully to ({rx}, {ry})."
            else:
                self.log_message = "Move backward FAILED! Obstacle detected or out of bounds."
        elif key == 'A':
            dir_name = self.service.turn_left()
            self.log_message = f"Turned left. Heading is now {dir_name.upper()}."
        elif key == 'D':
            dir_name = self.service.turn_right()
            self.log_message = f"Turned right. Heading is now {dir_name.upper()}."

        # Update labels and redraw
        self.pos_label.config(text=f"Position: ({self.service.get_robot_x()}, {self.service.get_robot_y()})")
        self.dir_label.config(text=f"Heading: {self.service.get_robot_direction_name().upper()}")
        self.log_label.config(text=self.log_message)
        self.draw_map()

    def get_canvas_coords(self, x, y):
        # Map coordinate space to canvas pixel space (flipping Y axis)
        padding = 60
        map_size = self.service.get_map_size()
        step = (self.canvas_size - 2 * padding) / (map_size - 1)
        cx = padding + x * step
        cy = self.canvas_size - (padding + y * step)
        return cx, cy

    def draw_map(self):
        self.canvas.delete("all")
        map_size = self.service.get_map_size()
        
        # Adjust dimensions dynamically based on grid density
        if map_size > 12:
            node_radius = 8
            cross_half_size = 4
            edge_width = 2
            robot_tip = 12
            robot_base = 8
            font_size = 6
        else:
            node_radius = 12
            cross_half_size = 6
            edge_width = 3
            robot_tip = 16
            robot_base = 12
            font_size = 8

        # 1. Draw Edges
        for e in self.service.get_edges():
            x1, y1 = e.src.x, e.src.y
            x2, y2 = e.dest.x, e.dest.y
            cx1, cy1 = self.get_canvas_coords(x1, y1)
            cx2, cy2 = self.get_canvas_coords(x2, y2)
            
            if e.isBlock == 1:
                # Check if this blocked edge has been discovered by the agent
                if self.service.is_edge_discovered((x1, y1), (x2, y2)):
                    # Discovered: Red dashed line
                    self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#ef5350", width=edge_width + 1, dash=(5, 5))
                else:
                    # Undiscovered: Yellow dashed line
                    self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#ffeb3b", width=edge_width + 1, dash=(5, 5))
            else:
                # Draw clear edges in gray lines
                self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#424242", width=edge_width)

        # 2. Draw Nodes
        for (x, y), node in self.service.get_nodes().items():
            cx, cy = self.get_canvas_coords(x, y)
            
            if node.isBlock == 1:
                # Check if this blocked node has been discovered by the agent
                if self.service.is_node_discovered((x, y)):
                    # Discovered: red filled circle with a white cross
                    self.canvas.create_oval(
                        cx - node_radius, cy - node_radius, 
                        cx + node_radius, cy + node_radius, 
                        fill="#ef5350", outline="#b71c1c", width=2
                    )
                    self.canvas.create_line(cx - cross_half_size, cy - cross_half_size, cx + cross_half_size, cy + cross_half_size, fill="#ffffff", width=2)
                    self.canvas.create_line(cx + cross_half_size, cy - cross_half_size, cx - cross_half_size, cy + cross_half_size, fill="#ffffff", width=2)
                else:
                    # Undiscovered: yellow filled circle with a dark cross
                    self.canvas.create_oval(
                        cx - node_radius, cy - node_radius, 
                        cx + node_radius, cy + node_radius, 
                        fill="#ffeb3b", outline="#f57f17", width=2
                    )
                    self.canvas.create_line(cx - cross_half_size, cy - cross_half_size, cx + cross_half_size, cy + cross_half_size, fill="#333333", width=2)
                    self.canvas.create_line(cx + cross_half_size, cy - cross_half_size, cx - cross_half_size, cy + cross_half_size, fill="#333333", width=2)
            else:
                # Clear node: dark blue/cyan glowing circle
                self.canvas.create_oval(
                    cx - node_radius, cy - node_radius, 
                    cx + node_radius, cy + node_radius, 
                    fill="#1f2937", outline="#00adb5", width=2
                )
                
            # Add small coordinate text above/below clear nodes for clarity (only if map is not too crowded)
            if map_size <= 18:
                offset_y = -(node_radius + 8)
                self.canvas.create_text(
                    cx, cy + offset_y, 
                    text=f"{x},{y}", 
                    fill="#888888", 
                    font=("Helvetica", font_size)
                )

        # 3. Draw Robot (Agent)
        rx, ry = self.service.get_robot_x(), self.service.get_robot_y()
        rcx, rcy = self.get_canvas_coords(rx, ry)
        dx, dy = self.service.get_robot_direction()
        
        # Calculate vertices of direction triangle pointing towards heading
        if (dx, dy) == (0, 1):    # UP
            points = [rcx, rcy - robot_tip, rcx - robot_base, rcy + robot_base, rcx + robot_base, rcy + robot_base]
        elif (dx, dy) == (1, 0):  # RIGHT
            points = [rcx + robot_tip, rcy, rcx - robot_base, rcy - robot_base, rcx - robot_base, rcy + robot_base]
        elif (dx, dy) == (0, -1): # DOWN
            points = [rcx, rcy + robot_tip, rcx + robot_base, rcy - robot_base, rcx - robot_base, rcy - robot_base]
        else:                     # LEFT
            points = [rcx - robot_tip, rcy, rcx + robot_base, rcy + robot_base, rcx + robot_base, rcy - robot_base]
            
        # Draw robot body
        self.canvas.create_polygon(points, fill="#fdd835", outline="#f57f17", width=2)
        # Draw a small center circle on robot
        self.canvas.create_oval(rcx - 3, rcy - 3, rcx + 3, rcy + 3, fill="#121212")
