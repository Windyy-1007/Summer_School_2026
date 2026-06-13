from src.map import LineMap

def print_map(map_obj):
    """
    Visualizes the LineMap in a clean, perfectly aligned text-based grid with a boxed frame.
    - Free nodes are shown as (x,y)
    - Blocked nodes are shown as [###]
    - Clear horizontal edges are shown as ---
    - Clear vertical edges are shown as |
    - Blocked edges are shown as x
    """
    coords = list(map_obj.nodes.keys())
    if not coords:
        print("Empty map.")
        return
    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)
    
    width = max_x + 1
    height = max_y + 1
    
    lines = []
    
    # Construct rows from top (max_y) down to 0
    for y in range(max_y, -1, -1):
        # 1. Node and Horizontal Edge Row
        node_row_parts = []
        for x in range(width):
            node = map_obj.nodes[(x, y)]
            if node.isBlock == 1:
                node_str = "[###]"
            else:
                node_str = f"({x},{y})"
            
            # Center the node string in a 7-character field
            node_row_parts.append(f"{node_str:^7}")
            
            if x < max_x:
                # Find horizontal edge between (x,y) and (x+1,y)
                edge = None
                for e in map_obj.edges:
                    if ((e.src.x, e.src.y) == (x, y) and (e.dest.x, e.dest.y) == (x+1, y)) or \
                       ((e.src.x, e.src.y) == (x+1, y) and (e.dest.x, e.dest.y) == (x, y)):
                        edge = e
                        break
                if edge and edge.isBlock == 1:
                    node_row_parts.append(" x ")
                else:
                    node_row_parts.append("---")
        lines.append("".join(node_row_parts))
        
        # 2. Vertical Edge Row
        if y > 0:
            vert_row_parts = []
            for x in range(width):
                # Find vertical edge between (x,y) and (x,y-1)
                edge = None
                for e in map_obj.edges:
                    if ((e.src.x, e.src.y) == (x, y) and (e.dest.x, e.dest.y) == (x, y-1)) or \
                       ((e.src.x, e.src.y) == (x, y-1) and (e.dest.x, e.dest.y) == (x, y)):
                        edge = e
                        break
                if edge:
                    if edge.isBlock == 1:
                        vert_row_parts.append(f"{'x':^7}")
                    else:
                        vert_row_parts.append(f"{'|':^7}")
                else:
                    vert_row_parts.append("       ")
                
                if x < max_x:
                    vert_row_parts.append("   ")
            lines.append("".join(vert_row_parts))
            
    # Calculate box borders dynamically
    row_len = width * 7 + (width - 1) * 3
    border_line = "+" + "-" * (row_len + 4) + "+"
    
    print("\n" + "=" * (row_len + 8))
    print(f" MAP VISUALIZATION ({width}x{height} Grid) ".center(row_len + 8, "="))
    print("=" * (row_len + 8))
    print("Legend: (x,y) = Clear Node,  [###] = Blocked Node")
    print("        --- or | = Clear Edge,   x   = Blocked Edge")
    print("-" * (row_len + 8))
    print(border_line)
    for line in lines:
        print(f"|  {line}  |")
    print(border_line)
    print("=" * (row_len + 8) + "\n")

def main():
    # 1. Initialize LineMap
    lm = LineMap()
    
    # 2. Create 4x4 Grid Map
    lm.createMap(4)
    
    # 3. Set obstacles
    # Block nodes (1, 1) and (2, 2)
    print("Setting blocked nodes at (1,1) and (2,2)...")
    lm.set_obstacles(node=(1, 1))
    lm.set_obstacles(node=(2, 2))
    
    # Block edge between (0,1) and (0,2)
    print("Setting blocked edge between (0,1) and (0,2)...")
    lm.set_obstacles(edge=((0, 1), (0, 2)))
    
    # Block edge between (2,0) and (3,0)
    print("Setting blocked edge between (2,0) and (3,0)...")
    lm.set_obstacles(edge=((2, 0), (3, 0)))

    # 4. Print Map
    print_map(lm)

if __name__ == "__main__":
    main()
