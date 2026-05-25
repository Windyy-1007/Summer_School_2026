
from abc import ABC, abstractmethod

class Node:
    def __init__ (self, x, y, isBlock = False):
        self.x = x
        self.y = y
        self.isBlock = isBlock

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_isBlock(self):
        return self.isBlock

    def set_Block(self, isBlock):
        self.isBlock = isBlock

class Edge:
    def __init__(self, src, dest, weight = 1, isBlock = False):
        self.src = src
        self.dest = dest
        self.weight = weight
        self.isBlock = isBlock
    
    def get_src(self):
        return self.src
    
    def get_dest(self):
        return self.dest
    
    def get_weight(self):
        return self.weight
    
    def set_weight(self, weight):
        self.weight = weight

class Map(ABC):
    def __init__(self):
        self.nodes = {}  # key = (x,y), value = Node
        self.edges = []  # list of Edge

    @abstractmethod
    def createMap(self, size):
        pass

    @abstractmethod
    def set_obstacles(self, node=None, edge=None):
        pass

class LineMap(Map):
    def __init__(self):
        super().__init__()

    def createMap(self, size):
        """
        Creates a square grid of size x size nodes.
        Nodes are located at coordinates (x, y) where x, y range from 0 to size - 1.
        The bottom-left node is (0, 0).
        Each node is connected to its 4 adjacent neighbors via directed edges.
        """
        if isinstance(size, int):
            width = height = size
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            width, height = size
        else:
            raise ValueError("Size must be an integer or a tuple/list of two integers.")

        self.nodes = {}
        self.edges = []

        # Create nodes
        for x in range(width):
            for y in range(height):
                self.nodes[(x, y)] = Node(x, y)

        # Create edges connecting to 4 adjacent neighbors
        for (x, y), node in self.nodes.items():
            # 4 adjacent neighbors: Up, Down, Left, Right
            neighbors = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                if (nx, ny) in self.nodes:
                    neighbor_node = self.nodes[(nx, ny)]
                    edge = Edge(node, neighbor_node)
                    self.edges.append(edge)

    def set_obstacles(self, node=None, edge=None):
        """
        Sets obstacle status to 1 (True) for the specified node or edge.
        
        Args:
            node: Node object or coordinate tuple (x, y) to block.
            edge: Edge object or tuple of two endpoints (either Node objects or coordinate tuples).
        """
        if node is not None:
            if isinstance(node, Node):
                node.isBlock = 1
            elif isinstance(node, tuple) and len(node) == 2:
                if node in self.nodes:
                    self.nodes[node].isBlock = 1
                else:
                    raise ValueError(f"Node at coordinate {node} does not exist in the map.")
            else:
                raise TypeError("node must be a Node object or a tuple of (x, y) coordinates.")

        if edge is not None:
            if isinstance(edge, Edge):
                edge.isBlock = 1
            elif isinstance(edge, tuple) and len(edge) == 2:
                item1, item2 = edge
                
                # Helper to get coordinate from Node or coordinate tuple
                def get_coord(item):
                    if isinstance(item, Node):
                        return (item.x, item.y)
                    elif isinstance(item, tuple) and len(item) == 2:
                        return item
                    else:
                        raise TypeError("Edge endpoints must be Node objects or coordinate tuples.")
                
                coord1 = get_coord(item1)
                coord2 = get_coord(item2)

                found_any = False
                for e in self.edges:
                    e_src_coord = (e.src.x, e.src.y)
                    e_dest_coord = (e.dest.x, e.dest.y)
                    # Block edge in both directions if they exist
                    if (e_src_coord == coord1 and e_dest_coord == coord2) or \
                       (e_src_coord == coord2 and e_dest_coord == coord1):
                        e.isBlock = 1
                        found_any = True
                
                if not found_any:
                    raise ValueError(f"No edge found between {coord1} and {coord2} in the map.")
            else:
                raise TypeError("edge must be an Edge object or a tuple of two endpoints.")

    # Alias to handle various spellings or singular/plural naming
    set_obstacle = set_obstacles