import tkinter as tk
from tkinter import filedialog

class RobotApp:
    MAX_CANVAS_SIZE = 750
    MIN_CANVAS_SIZE = 240
    WINDOW_MARGIN = 80
    CONTROL_HEIGHT_ESTIMATE = 350

    def __init__(self, root, service):
        self.root = root
        self.service = service
        
        self.root.title("LineMap Robot Simulator")
        self._configure_window_size()
        self.root.configure(bg="#1a1a1a")
        
        self.log_message = "Simulator started. Use A W S D to control."
        self.log_lines = [self.log_message]
        self.active_path = None
        self.active_path_color = "#29b6f6"
        self.is_auto_running = False
        self.is_training_running = False
        self.edit_mode = None
        self.pending_edge_start = None
        self._resize_after_id = None
        
        # Build GUI
        self.setup_ui()
        self.root.bind("<Configure>", self.schedule_resize)
        
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

    def _configure_window_size(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(940, max(560, screen_width - self.WINDOW_MARGIN))
        height = min(1100, max(600, screen_height - self.WINDOW_MARGIN))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(520, 560)
        self.root.resizable(True, True)
        self.canvas_size = self._canvas_size_for_window(width, height)

    def _canvas_size_for_window(self, width, height):
        available_width = max(self.MIN_CANVAS_SIZE, width - self.WINDOW_MARGIN)
        available_height = max(
            self.MIN_CANVAS_SIZE,
            height - self.CONTROL_HEIGHT_ESTIMATE,
        )
        return int(
            max(
                self.MIN_CANVAS_SIZE,
                min(self.MAX_CANVAS_SIZE, available_width, available_height),
            )
        )

    def schedule_resize(self, event):
        if event.widget is not self.root or not hasattr(self, "canvas"):
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after_idle(self.apply_responsive_canvas_size)

    def apply_responsive_canvas_size(self):
        self._resize_after_id = None
        new_size = self._canvas_size_for_window(
            self.root.winfo_width(),
            self.root.winfo_height(),
        )
        if new_size == self.canvas_size:
            return
        self.canvas_size = new_size
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)
        self.draw_map()

    def setup_ui(self):
        self.scroll_frame = tk.Frame(self.root, bg="#1a1a1a")
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(
            self.scroll_frame,
            bg="#1a1a1a",
            highlightthickness=0,
            bd=0
        )
        self.scrollbar = tk.Scrollbar(
            self.scroll_frame,
            orient=tk.VERTICAL,
            command=self.scroll_canvas.yview
        )
        self.content_frame = tk.Frame(self.scroll_canvas, bg="#1a1a1a")
        self.content_window = self.scroll_canvas.create_window(
            (0, 0),
            window=self.content_frame,
            anchor="nw"
        )

        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.content_frame.bind("<Configure>", self.update_scroll_region)
        self.scroll_canvas.bind("<Configure>", self.resize_scroll_content)
        self.root.bind_all("<MouseWheel>", self.handle_mousewheel)
        self.root.bind_all("<Button-4>", self.handle_mousewheel)
        self.root.bind_all("<Button-5>", self.handle_mousewheel)

        parent = self.content_frame

        # Header Label
        header = tk.Label(
            parent,
            text="ROBOT NAVIGATION SIMULATOR", 
            font=("Helvetica", 16, "bold"), 
            bg="#1a1a1a", 
            fg="#ffffff", 
            pady=15
        )
        header.pack()

        self.map_label = tk.Label(
            parent,
            text=f"Map: {self.service.get_map_name()}",
            font=("Helvetica", 11, "bold"),
            bg="#1a1a1a",
            fg="#90caf9",
        )
        self.map_label.pack(pady=(0, 4))

        # Canvas for Drawing Map
        self.canvas = tk.Canvas(
            parent,
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg="#262626", 
            highlightthickness=0
        )
        self.canvas.pack(pady=(4, 8))

        # Status and Control Frame
        info_frame = tk.Frame(parent, bg="#1a1a1a", pady=10)
        info_frame.pack(fill=tk.X, padx=20)

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

        # Primary actions
        button_frame = tk.Frame(parent, bg="#1a1a1a")
        button_frame.pack(fill=tk.X, padx=20, pady=5)
        for column in range(3):
            button_frame.columnconfigure(column, weight=1, uniform="planner")

        self.astar_button = tk.Button(
            button_frame,
            text="Show A* Reference",
            command=self.show_astar_path,
            bg="#263238",
            fg="#ffffff",
            activebackground="#37474f",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.astar_button.grid(row=0, column=0, sticky="ew", padx=5, pady=3)

        self.student_rl_run_button = tk.Button(
            button_frame,
            text="Run Student RL",
            command=self.run_student_rl_policy,
            bg="#ad1457",
            fg="#ffffff",
            activebackground="#d81b60",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.student_rl_run_button.grid(row=0, column=1, sticky="ew", padx=5, pady=3)

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
        self.reset_button.grid(row=0, column=2, sticky="ew", padx=5, pady=3)

        edit_frame = tk.LabelFrame(
            parent,
            text="Map editor",
            bg="#1a1a1a",
            fg="#bdbdbd",
            padx=5,
            pady=5,
        )
        edit_frame.pack(fill=tk.X, padx=20, pady=5)

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

        map_io_frame = tk.Frame(parent, bg="#1a1a1a")
        map_io_frame.pack(fill=tk.X, padx=20, pady=5)

        self.save_map_button = tk.Button(
            map_io_frame,
            text="Save Map JSON",
            command=self.save_map_dialog,
            bg="#263238",
            fg="#ffffff",
            activebackground="#37474f",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.save_map_button.pack(side=tk.LEFT, padx=5)

        self.load_map_button = tk.Button(
            map_io_frame,
            text="Load Map JSON",
            command=self.load_map_dialog,
            bg="#263238",
            fg="#ffffff",
            activebackground="#37474f",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.load_map_button.pack(side=tk.LEFT, padx=5)

        training_frame = tk.LabelFrame(
            parent,
            text="Student RL training",
            bg="#1a1a1a",
            fg="#bdbdbd",
            padx=5,
            pady=5,
        )
        training_frame.pack(fill=tk.X, padx=20, pady=5)
        for column in range(6):
            training_frame.columnconfigure(column, weight=1 if column in (2, 3, 4, 5) else 0)

        self.training_map_var = tk.IntVar(value=1)
        training_count = max(1, len(self.service.get_training_map_files()))
        tk.Label(
            training_frame,
            text="Map",
            bg="#1a1a1a",
            fg="#ffffff",
        ).grid(row=0, column=0, padx=(5, 2), pady=3)
        self.training_map_spinbox = tk.Spinbox(
            training_frame,
            from_=1,
            to=training_count,
            width=4,
            textvariable=self.training_map_var,
            bg="#263238",
            fg="#ffffff",
            buttonbackground="#37474f",
            relief=tk.FLAT
        )
        self.training_map_spinbox.grid(row=0, column=1, padx=3, pady=3)

        self.load_training_button = tk.Button(
            training_frame,
            text="Load Training Map",
            command=self.load_training_map,
            bg="#1e3a5f",
            fg="#ffffff",
            activebackground="#244d7a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.load_training_button.grid(row=0, column=2, sticky="ew", padx=3, pady=3)

        self.training_episodes_var = tk.IntVar(value=500)
        tk.Label(
            training_frame,
            text="Episodes per map",
            bg="#1a1a1a",
            fg="#ffffff",
        ).grid(row=1, column=0, padx=(5, 2), pady=3)
        self.training_episodes_spinbox = tk.Spinbox(
            training_frame,
            from_=50,
            to=10000,
            increment=50,
            width=6,
            textvariable=self.training_episodes_var,
            bg="#263238",
            fg="#ffffff",
            buttonbackground="#37474f",
            relief=tk.FLAT
        )
        self.training_episodes_spinbox.grid(row=1, column=1, padx=3, pady=3)

        self.train_rl_button = tk.Button(
            training_frame,
            text="Train Current Map",
            command=self.run_student_rl_training,
            bg="#6a1b9a",
            fg="#ffffff",
            activebackground="#8e24aa",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.train_rl_button.grid(row=1, column=2, sticky="ew", padx=3, pady=3)

        self.train_all_rl_button = tk.Button(
            training_frame,
            text="Train All Maps",
            command=self.train_student_rl_all_maps,
            bg="#ad1457",
            fg="#ffffff",
            activebackground="#d81b60",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.train_all_rl_button.grid(row=1, column=3, sticky="ew", padx=3, pady=3)

        self.stop_rl_button = tk.Button(
            training_frame,
            text="Stop RL",
            command=self.stop_student_rl_training,
            bg="#4e342e",
            fg="#ffffff",
            activebackground="#6d4c41",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.stop_rl_button.grid(row=1, column=4, sticky="ew", padx=3, pady=3)

        self.reset_rl_policy_button = tk.Button(
            training_frame,
            text="Reset RL Model",
            command=self.reset_student_rl_policy,
            bg="#7f1d1d",
            fg="#ffffff",
            activebackground="#991b1b",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6
        )
        self.reset_rl_policy_button.grid(row=1, column=5, sticky="ew", padx=3, pady=3)

        self.training_progress_label = tk.Label(
            training_frame,
            text=self.service.get_student_rl_training_progress(),
            font=("Consolas", 10, "bold"),
            bg="#1a1a1a",
            fg="#ffca28",
            anchor="w",
        )
        self.training_progress_label.grid(
            row=2,
            column=0,
            columnspan=6,
            sticky="ew",
            padx=5,
            pady=(5, 2),
        )

        # Console Log Box
        log_frame = tk.Frame(parent, bg="#1a1a1a")
        log_frame.pack(fill=tk.X, padx=20, pady=5)
        
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
            parent,
            text="Controls: W (Forward) | S (Backward) | A (Turn Left) | D (Turn Right)", 
            font=("Helvetica", 10, "italic"), 
            bg="#1a1a1a", 
            fg="#888888", 
            pady=10
        )
        guide.pack()

    def update_scroll_region(self, event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def resize_scroll_content(self, event):
        self.scroll_canvas.itemconfigure(self.content_window, width=event.width)
        self.update_scroll_region()

    def handle_mousewheel(self, event):
        if not hasattr(self, "scroll_canvas"):
            return
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.scroll_canvas.yview_scroll(delta, "units")

    def save_map_dialog(self):
        self.is_auto_running = False
        self.is_training_running = False
        file_path = filedialog.asksaveasfilename(
            initialdir=self.service.get_custom_map_dir(),
            defaultextension=".json",
            filetypes=[("JSON maps", "*.json"), ("All files", "*.*")],
            title="Save current map",
        )
        if not file_path:
            return
        try:
            self.log_message = self.service.save_map(file_path)
        except Exception as exc:
            self.log_message = f"Save failed: {exc}"
        self.update_status()
        self.draw_map()

    def load_map_dialog(self):
        self.is_auto_running = False
        self.is_training_running = False
        file_path = filedialog.askopenfilename(
            initialdir=self.service.get_custom_map_dir(),
            filetypes=[("JSON maps", "*.json"), ("All files", "*.*")],
            title="Load map",
        )
        if not file_path:
            return
        try:
            self.log_message = self.service.load_map(file_path)
            self.active_path = None
            self.pending_edge_start = None
        except Exception as exc:
            self.log_message = f"Load failed: {exc}"
        self.update_status()
        self.draw_map()

    def load_training_map(self):
        self.is_auto_running = False
        self.is_training_running = False
        file_path, message = self.service.load_training_map(self.training_map_var.get())
        self.active_path = None
        self.pending_edge_start = None
        self.log_message = message
        self.update_status()
        self.draw_map()

    def run_student_rl_training(self):
        if self.is_training_running:
            return
        self.is_auto_running = False
        self.pending_edge_start = None
        self.active_path = None
        self.active_path_color = "#ffca28"
        self.is_training_running = True
        episodes = self.training_episodes_var.get()
        self.log_message = self.service.start_student_rl_training(
            episodes=episodes,
            batch_size=25,
        )
        self.update_status()
        self.draw_map()
        self.root.after(100, self.animate_student_rl_training)

    def train_student_rl_all_maps(self):
        if self.is_training_running:
            return
        self.is_auto_running = False
        self.is_training_running = True
        self.pending_edge_start = None
        self.active_path = None
        self.active_path_color = "#ffca28"
        episodes = self.training_episodes_var.get()
        self.log_message = self.service.start_student_rl_all_training_maps(
            episodes=episodes,
            batch_size=25,
        )
        self.update_status()
        self.draw_map()
        if not self.service.is_student_rl_training_active():
            self.is_training_running = False
            return
        self.root.after(50, self.animate_student_rl_training)

    def animate_student_rl_training(self):
        if not self.is_training_running:
            return
        try:
            done, message, path = self.service.train_student_rl_batch()
        except Exception as exc:
            self.is_training_running = False
            self.service.stop_student_rl_training()
            self.log_message = f"Student RL training failed: {exc}"
            self.update_status()
            return
        self.active_path = path
        self.active_path_color = "#ffca28"
        self.log_message = message
        self.update_status()
        self.draw_map()
        if done:
            self.is_training_running = False
        else:
            self.root.after(10, self.animate_student_rl_training)

    def stop_student_rl_training(self):
        self.is_training_running = False
        self.log_message = self.service.stop_student_rl_training()
        self.update_status()
        self.draw_map()

    def reset_student_rl_policy(self):
        self.is_auto_running = False
        self.is_training_running = False
        self.active_path = None
        self.pending_edge_start = None
        self.log_message = self.service.reset_student_rl_policy()
        self.update_status()
        self.draw_map()

    def handle_action(self, key):
        self.is_auto_running = False
        self.is_training_running = False
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
        if not self.log_lines or self.log_lines[-1] != self.log_message:
            self.log_lines.append(self.log_message)
        self.log_lines = self.log_lines[-5:]
        self.pos_label.config(text=f"Position: ({self.service.get_robot_x()}, {self.service.get_robot_y()})")
        self.dir_label.config(text=f"Heading: {self.service.get_robot_direction_name().upper()}")
        self.score_label.config(text=f"Score: {self.service.get_score()}")
        self.map_label.config(text=f"Map: {self.service.get_map_name()}")
        self.training_progress_label.config(
            text=self.service.get_student_rl_training_progress()
        )
        self.log_label.config(text="\n".join(self.log_lines))

    def set_edit_mode(self, mode):
        self.is_auto_running = False
        self.is_training_running = False
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
        padding = self.get_canvas_padding()
        map_width = self.service.get_map_width()
        map_height = self.service.get_map_height()
        step_x = (self.canvas_size - 2 * padding) / max(1, map_width - 1)
        step_y = (self.canvas_size - 2 * padding) / max(1, map_height - 1)
        x = round((event.x - padding) / step_x)
        y = round((self.canvas_size - event.y - padding) / step_y)
        coord = (x, y)
        if coord not in self.service.get_nodes():
            return None
        cx, cy = self.get_canvas_coords(x, y)
        threshold = min(step_x, step_y) * 0.35
        if abs(event.x - cx) > threshold or abs(event.y - cy) > threshold:
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
        self.is_training_running = False
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
        self.is_training_running = False
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
        self.is_training_running = False
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
        self.is_training_running = False
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

    def run_student_rl_policy(self):
        if self.is_auto_running:
            return
        self.is_training_running = False
        self.service.reset_robot()
        self.pending_edge_start = None
        self.active_path = None
        self.is_auto_running = True
        self.log_message = f"Running saved Student RL policy toward {self.service.get_goal()}."
        self.update_status()
        self.draw_map()
        self.root.after(350, self.animate_student_rl_policy_step)

    def animate_student_rl_policy_step(self):
        if not self.is_auto_running:
            return

        status, message, path = self.service.step_toward_goal_with_known_student_rl()
        self.active_path = path
        self.active_path_color = "#ffca28"
        self.log_message = message
        self.update_status()
        self.draw_map()

        if status is True:
            self.is_auto_running = False
            self.log_message = f"Student RL policy reached {self.service.get_goal()}."
            self.update_status()
        elif status is False:
            self.is_auto_running = False
        else:
            self.root.after(350, self.animate_student_rl_policy_step)

    def reset_robot(self):
        self.is_auto_running = False
        self.is_training_running = False
        self.service.reset_robot()
        self.active_path = None
        self.pending_edge_start = None
        self.log_message = "Robot reset to start."
        self.update_status()
        self.draw_map()

    def get_canvas_coords(self, x, y):
        # Map coordinate space to canvas pixel space (flipping Y axis)
        padding = self.get_canvas_padding()
        map_width = self.service.get_map_width()
        map_height = self.service.get_map_height()
        step_x = (self.canvas_size - 2 * padding) / max(1, map_width - 1)
        step_y = (self.canvas_size - 2 * padding) / max(1, map_height - 1)
        cx = padding + x * step_x
        cy = self.canvas_size - (padding + y * step_y)
        return cx, cy

    def get_canvas_padding(self):
        return max(24, min(60, self.canvas_size * 0.08))

    def draw_map(self):
        self.canvas.delete("all")
        map_size = self.service.get_map_size()
        padding = self.get_canvas_padding()
        map_width = self.service.get_map_width()
        map_height = self.service.get_map_height()
        step_x = (self.canvas_size - 2 * padding) / max(1, map_width - 1)
        step_y = (self.canvas_size - 2 * padding) / max(1, map_height - 1)
        spacing = max(1, min(step_x, step_y))
        
        # Adjust dimensions dynamically based on both grid density and canvas size.
        node_radius = max(4, min(12, spacing * 0.24))
        cross_half_size = max(2, node_radius * 0.45)
        edge_width = max(1, int(min(3, spacing / 10)))
        robot_tip = max(8, node_radius * 1.5)
        robot_base = max(6, node_radius)
        font_size = max(5, int(min(8, spacing * 0.25)))

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
            if map_size <= 18 and spacing >= 20:
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
