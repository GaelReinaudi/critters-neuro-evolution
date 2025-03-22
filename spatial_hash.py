"""
Spatial hash grid for efficient collision detection and neighbor finding
"""
import math

class SpatialHash:
    """
    A spatial hash grid for efficient proximity queries
    """
    def __init__(self, cell_size=100):
        """
        Initialize the spatial hash grid
        
        Args:
            cell_size (float): Size of each grid cell
        """
        self.cell_size = cell_size
        self.grid = {}  # Maps grid cell (x,y) to list of entities
        
    def _get_cell_pos(self, x, y):
        """
        Get the grid cell coordinates for a world position
        
        Args:
            x (float): World x-coordinate
            y (float): World y-coordinate
            
        Returns:
            tuple: Grid cell coordinates (cell_x, cell_y)
        """
        cell_x = math.floor(x / self.cell_size)
        cell_y = math.floor(y / self.cell_size)
        return (cell_x, cell_y)
    
    def _get_cells_for_entity(self, entity):
        """
        Get all grid cells that an entity might overlap with
        
        Args:
            entity: Entity with x, y, and size attributes
            
        Returns:
            list: List of grid cell coordinates
        """
        # For simplicity, just return the cell containing the entity's center
        return [self._get_cell_pos(entity.x, entity.y)]
    
    def clear(self):
        """Clear all entities from the grid"""
        self.grid = {}
        
    def insert(self, entity):
        """
        Insert an entity into the grid
        
        Args:
            entity: Entity with x, y, and size attributes
        """
        cell_positions = self._get_cells_for_entity(entity)
        for cell_pos in cell_positions:
            if cell_pos not in self.grid:
                self.grid[cell_pos] = []
            self.grid[cell_pos].append(entity)
            
    def get_nearby_entities(self, entity, max_distance=None):
        """
        Get entities near the given entity
        
        Args:
            entity: Entity with x, y, and size attributes
            max_distance (float, optional): Maximum distance to consider. If None, cell size is used.
            
        Returns:
            list: List of nearby entities, excluding the given entity
        """
        if max_distance is None:
            max_distance = self.cell_size
            
        # Get all potential nearby entities from the grid
        nearby = []
        cell_positions = self._get_cells_for_entity(entity)
        
        # Also check neighboring cells
        neighbor_offsets = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), 
                           (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        for cell_pos in cell_positions:
            cell_x, cell_y = cell_pos
            for offset_x, offset_y in neighbor_offsets:
                neighbor_pos = (cell_x + offset_x, cell_y + offset_y)
                if neighbor_pos in self.grid:
                    nearby.extend(self.grid[neighbor_pos])
        
        # Filter out the entity itself and duplicates
        unique_nearby = []
        for other in nearby:
            if other is not entity and other not in unique_nearby:
                # Optional distance filter
                if max_distance:
                    dx = other.x - entity.x
                    dy = other.y - entity.y
                    if dx*dx + dy*dy <= max_distance*max_distance:
                        unique_nearby.append(other)
                else:
                    unique_nearby.append(other)
                    
        return unique_nearby 