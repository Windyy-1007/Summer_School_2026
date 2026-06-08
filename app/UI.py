import tkinter as tk

class RobotApp:
    def __init__(self, root, service):
        self.root = root
        self.service = service
        
        self.root.title("LineMap Robot Simulator")
        self.root.geometry("900x980")
        self.root.configure(bg="#1a1a1a")
        
        self.log_message = "Simulator started. Use A W S D to control."
        self.active_path = None
        self.active_path_color = "#29b6f6"
        self.is_auto_running = False
        self.edit_mode = None
        self.pending_edge_start = None
        
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
        self.canvas.bind("<Button-1>", self.handle_canvas_click)
        
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

        self.score_label = tk.Label(
            info_frame,
            text=f"Score: {self.service.get_score()}",
            font=("Helvetica", 12, "bold"),
            bg="#1a1a1a",
            fg="#ffd54f"
        )
        self.score_label.pack(side=tk.LEFT, padx=30)

        # Planner Buttons
        button_frame = tk.Frame(self.root, bg="#1a1a1a")
        button_frame.pack(fill=tk.X, padx=40, pady=5)

        self.astar_button = tk.Button(
            button_frame,
            text="Show A* Path",
            command=self.show_astar_path,
            bg="#263238",
            fg="#ffffff",
            activebackground="#37474f",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.astar_button.pack(side=tk.LEFT, padx=5)

        self.qlearning_button = tk.Button(
            button_frame,
            text="Show Q-learning Path",
            command=self.show_qlearning_path,
            bg="#263238",
            fg="#ffffff",
            activebackground="#37474f",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.qlearning_button.pack(side=tk.LEFT, padx=5)

        self.two_map_button = tk.Button(
            button_frame,
            text="Run Two-map A*",
            command=self.run_two_map_astar,
            bg="#00695c",
            fg="#ffffff",
            activebackground="#00897b",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.two_map_button.pack(side=tk.LEFT, padx=5)

        self.two_map_q_button = tk.Button(
            button_frame,
            text="Run Two-map Q",
            command=self.run_two_map_qlearning,
            bg="#6a1b9a",
            fg="#ffffff",
            activebackground="#8e24aa",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.two_map_q_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(
            button_frame,
            text="Reset Robot",
            command=self.reset_robot,
            bg="#4e342e",
            fg="#ffffff",
            activebackground="#6d4c41",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.reset_button.pack(side=tk.RIGHT, padx=5)

        edit_frame = tk.Frame(self.root, bg="#1a1a1a")
        edit_frame.pack(fill=tk.X, padx=40, pady=5)

        self.goal_button = tk.Button(
            edit_frame,
            text="Set Goal",
            command=lambda: self.set_edit_mode("goal"),
            bg="#1e3a5f",
            fg="#ffffff",
            activebackground="#244d7a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.goal_button.pack(side=tk.LEFT, padx=5)

        self.sub_goal_button = tk.Button(
            edit_frame,
            text="Toggle Sub-goal",
            command=lambda: self.set_edit_mode("sub_goal"),
            bg="#4a148c",
            fg="#ffffff",
            activebackground="#6a1b9a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.sub_goal_button.pack(side=tk.LEFT, padx=5)

        self.node_button = tk.Button(
            edit_frame,
            text="Toggle Node Obstacle",
            command=lambda: self.set_edit_mode("node"),
            bg="#5d4037",
            fg="#ffffff",
            activebackground="#795548",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.node_button.pack(side=tk.LEFT, padx=5)

        self.edge_button = tk.Button(
            edit_frame,
            text="Toggle Edge Obstacle",
            command=lambda: self.set_edit_mode("edge"),
            bg="#5d4037",
            fg="#ffffff",
            activebackground="#795548",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.edge_button.pack(side=tk.LEFT, padx=5)

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
        self.is_auto_running = False
        self.active_path = None
        self.pending_edge_start = None

        if key == 'W':
            success, score_message = self.service.forward()
            rx, ry = self.service.get_robot_x(), self.service.get_robot_y()
            if success:
                self.log_message = f"Moved forward successfully to ({rx}, {ry}).{score_message}"
            else:
                self.log_message = f"Move forward FAILED! Obstacle detected or out of bounds.{score_message}"
        elif key == 'S':
            success, score_message = self.service.backward()
            rx, ry = self.service.get_robot_x(), self.service.get_robot_y()
            if success:
                self.log_message = f"Moved backward successfully to ({rx}, {ry}).{score_message}"
            else:
                self.log_message = f"Move backward FAILED! Obstacle detected or out of bounds.{score_message}"
        elif key == 'A':
            dir_name, score_message = self.service.turn_left()
            self.log_message = f"Turned left. Heading is now {dir_name.upper()}.{score_message}"
        elif key == 'D':
            dir_name, score_message = self.service.turn_right()
            self.log_message = f"Turned right. Heading is now {dir_name.upper()}.{score_message}"

        # Update labels and redraw
        self.pos_label.config(text=f"Position: ({self.service.get_robot_x()}, {self.service.get_robot_y()})")
        self.dir_label.config(text=f"Heading: {self.service.get_robot_direction_name().upper()}")
        self.score_label.config(text=f"Score: {self.service.get_score()}")
        self.log_label.config(text=self.log_message)
        self.draw_map()

    def update_status(self):
        self.pos_label.config(text=f"Position: ({self.service.get_robot_x()}, {self.service.get_robot_y()})")
        self.dir_label.config(text=f"Heading: {self.service.get_robot_direction_name().upper()}")
        self.score_label.config(text=f"Score: {self.service.get_score()}")
        self.log_label.config(text=self.log_message)

    def set_edit_mode(self, mode):
        self.is_auto_running = False
        self.active_path = None
        self.pending_edge_start = None
        self.edit_mode = mode
        labels = {
            "goal": "Click a node to set the goal.",
            "sub_goal": "Click a node to add or remove a sub-goal.",
            "node": "Click a node to toggle a node obstacle.",
            "edge": "Click two adjacent nodes to toggle an edge obstacle."
        }
        self.log_message = labels[mode]
        self.update_status()
        self.draw_map()

    def get_coord_from_canvas(self, event):
        padding = 60
        map_size = self.service.get_map_size()
        step = (self.canvas_size - 2 * padding) / (map_size - 1)
        x = round((event.x - padding) / step)
        y = round((self.canvas_size - event.y - padding) / step)
        coord = (x, y)
        if coord not in self.service.get_nodes():
            return None
        cx, cy = self.get_canvas_coords(x, y)
        if abs(event.x - cx) > step * 0.35 or abs(event.y - cy) > step * 0.35:
            return None
        return coord

    def handle_canvas_click(self, event):
        if self.edit_mode is None:
            return

        coord = self.get_coord_from_canvas(event)
        if coord is None:
            self.log_message = "Click closer to a map node."
        elif self.edit_mode == "goal":
            if self.service.set_goal(coord):
                self.log_message = f"Goal set to {coord}. Robot reset."
            else:
                self.log_message = "Goal must be a valid node and cannot be the start."
        elif self.edit_mode == "sub_goal":
            added = self.service.toggle_sub_goal(coord)
            if added is None:
                self.log_message = "Sub-goal must be a valid node and cannot be start or final goal."
            elif added:
                self.log_message = f"Sub-goal added at {coord}. Robot reset."
            else:
                self.log_message = f"Sub-goal removed at {coord}. Robot reset."
        elif self.edit_mode == "node":
            blocked = self.service.toggle_node_obstacle(coord)
            if blocked is None:
                self.log_message = "Cannot block the start, goal, or sub-goal nodes."
            elif blocked:
                self.log_message = f"Node obstacle added at {coord}. Robot reset."
            else:
                self.log_message = f"Node obstacle removed at {coord}. Robot reset."
        elif self.edit_mode == "edge":
            if self.pending_edge_start is None:
                self.pending_edge_start = coord
                self.log_message = f"Selected edge start {coord}. Click an adjacent node."
            else:
                blocked = self.service.toggle_edge_obstacle(self.pending_edge_start, coord)
                if blocked is None:
                    self.log_message = "Second node must be adjacent to the first."
                elif blocked:
                    self.log_message = f"Edge obstacle added: {self.pending_edge_start} <-> {coord}. Robot reset."
                else:
                    self.log_message = f"Edge obstacle removed: {self.pending_edge_start} <-> {coord}. Robot reset."
                self.pending_edge_start = None

        self.active_path = None
        self.update_status()
        self.draw_map()

    def show_astar_path(self):
        self.is_auto_running = False
        self.active_path = self.service.get_astar_path(use_known_map=False)
        self.active_path_color = "#29b6f6"
        if self.active_path:
            self.log_message = f"A* path shown: {len(self.active_path)} nodes through sub-goals to {self.service.get_goal()}."
        else:
            self.log_message = "A* could not find a path."
        self.update_status()
        self.draw_map()

    def show_qlearning_path(self):
        self.is_auto_running = False
        self.pending_edge_start = None
        self.active_path = self.service.get_qlearning_path(from_robot=False)
        self.active_path_color = "#ab47bc"
        if self.active_path:
            self.log_message = f"Q-learning path shown: {len(self.active_path)} nodes through sub-goals to {self.service.get_goal()}."
        else:
            self.log_message = "Q-learning could not find a path. Try more episodes in services.py."
        self.update_status()
        self.draw_map()

    def run_two_map_astar(self):
        if self.is_auto_running:
            return
        self.service.reset_robot()
        self.pending_edge_start = None
        self.active_path = None
        self.is_auto_running = True
        self.log_message = f"Running two-map A* toward {self.service.get_goal()}."
        self.update_status()
        self.draw_map()
        self.root.after(350, self.animate_two_map_step)

    def animate_two_map_step(self):
        if not self.is_auto_running:
            return

        self.active_path = self.service.get_astar_path(use_known_map=True)
        self.active_path_color = "#66bb6a"
        status, message = self.service.step_toward_goal_with_known_astar()
        self.log_message = message
        self.update_status()
        self.draw_map()

        if status is True:
            self.is_auto_running = False
            self.log_message = f"Two-map A* reached {self.service.get_goal()}."
            self.update_status()
        elif status is False:
            self.is_auto_running = False
        else:
            self.root.after(350, self.animate_two_map_step)

    def run_two_map_qlearning(self):
        if self.is_auto_running:
            return
        self.service.reset_robot()
        self.pending_edge_start = None
        self.active_path = None
        self.is_auto_running = True
        self.log_message = f"Running two-map Q-learning toward {self.service.get_goal()}."
        self.update_status()
        self.draw_map()
        self.root.after(350, self.animate_two_map_qlearning_step)

    def animate_two_map_qlearning_step(self):
        if not self.is_auto_running:
            return

        status, message, path = self.service.step_toward_goal_with_known_qlearning()
        self.active_path = path
        self.active_path_color = "#ab47bc"
        self.log_message = message
        self.update_status()
        self.draw_map()

        if status is True:
            self.is_auto_running = False
            self.log_message = f"Two-map Q-learning reached {self.service.get_goal()}."
            self.update_status()
        elif status is False:
            self.is_auto_running = False
        else:
            self.root.after(350, self.animate_two_map_qlearning_step)

    def reset_robot(self):
        self.is_auto_running = False
        self.service.reset_robot()
        self.active_path = None
        self.pending_edge_start = None
        self.log_message = "Robot reset to start."
        self.update_status()
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
        if self.active_path and len(self.active_path) > 1:
            path_points = []
            for x, y in self.active_path:
                cx, cy = self.get_canvas_coords(x, y)
                path_points.extend([cx, cy])
            self.canvas.create_line(
                *path_points,
                fill=self.active_path_color,
                width=edge_width + 4,
                smooth=True,
                splinesteps=1
            )

        start_x, start_y = self.service.get_start()
        goal_x, goal_y = self.service.get_goal()
        sub_goals = set(self.service.get_sub_goals())
        if self.pending_edge_start:
            pcx, pcy = self.get_canvas_coords(*self.pending_edge_start)
            self.canvas.create_oval(
                pcx - node_radius - 5,
                pcy - node_radius - 5,
                pcx + node_radius + 5,
                pcy + node_radius + 5,
                outline="#ffffff",
                width=2
            )

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
                fill = "#1f2937"
                outline = "#00adb5"
                if (x, y) == (start_x, start_y):
                    fill = "#2e7d32"
                    outline = "#a5d6a7"
                elif (x, y) == (goal_x, goal_y):
                    fill = "#1565c0"
                    outline = "#90caf9"
                elif (x, y) in sub_goals:
                    if self.service.is_sub_goal_reached((x, y)):
                        fill = "#6d4c41"
                        outline = "#ffe082"
                    else:
                        fill = "#4a148c"
                        outline = "#ce93d8"
                self.canvas.create_oval(
                    cx - node_radius, cy - node_radius, 
                    cx + node_radius, cy + node_radius, 
                    fill=fill, outline=outline, width=2
                )

                if (x, y) in sub_goals:
                    self.canvas.create_text(
                        cx,
                        cy,
                        text="S",
                        fill="#ffffff",
                        font=("Helvetica", font_size + 2, "bold")
                    )
                elif (x, y) == (goal_x, goal_y):
                    self.canvas.create_text(
                        cx,
                        cy,
                        text="G",
                        fill="#ffffff",
                        font=("Helvetica", font_size + 2, "bold")
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
