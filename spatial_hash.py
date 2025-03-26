"""
Spatial hash grid for efficient entity collision detection
"""
from config import CELL_SIZE

class SpatialHash:
    """
    A spatial hash grid for efficiently finding nearby entities
    """
    def __init__(self, cell_size=CELL_SIZE):
        """
        Initialize the spatial hash grid
        
        Args:
            cell_size (int): Size of each grid cell
        """
        self.cell_size = cell_size
        self.grid = {}  # Dictionary mapping cell positions to lists of entities
        
    def _get_cell_pos(self, x, y):
        """
        Get the grid cell coordinates for a world position
        
        Args:
            x (float): World x-coordinate
            y (float): World y-coordinate
            
        Returns:
            tuple: Grid cell coordinates (cell_x, cell_y)
        """
        cell_x = int(x // self.cell_size)
        cell_y = int(y // self.cell_size)
        return (cell_x, cell_y)
    
    def _get_cells_for_entity(self, entity, radius=None):
        """
        Get all grid cells that an entity overlaps with
        
        Args:
            entity: Entity with x, y attributes
            radius: Optional extra radius to consider around the entity
            
        Returns:
            list: List of cell positions (tuples)
        """
        # Handle entities with size attribute
        entity_radius = getattr(entity, 'size', 0)
        
        # Add the search radius if provided
        if radius is not None:
            entity_radius += radius
            
        # Determine cell range to check
        min_x = entity.x - entity_radius
        max_x = entity.x + entity_radius
        min_y = entity.y - entity_radius
        max_y = entity.y + entity_radius
        
        # Convert to cell positions
        min_cell_x = int(min_x // self.cell_size)
        max_cell_x = int(max_x // self.cell_size)
        min_cell_y = int(min_y // self.cell_size)
        max_cell_y = int(max_y // self.cell_size)
        
        # Get all cells in the rectangle
        cells = [(x, y) 
                for x in range(min_cell_x, max_cell_x + 1)
                for y in range(min_cell_y, max_cell_y + 1)]
                
        return cells
    
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
            
    def get_nearby_entities(self, entity, radius):
        """
        Get all entities within a certain radius of the given entity
        
        Args:
            entity: Entity to check around
            radius: Maximum distance to search
            
        Returns:
            list: Entities within the radius
        """
        # Get all potential cells that could contain entities within radius
        cells = self._get_cells_for_entity(entity)
        
        # Create a set for faster lookups
        nearby_entities = set()
        
        # Entity's position for distance calculations
        ex, ey = entity.x, entity.y
        radius_squared = radius * radius
        
        # For each cell, check entities
        for cell in cells:
            if cell in self.grid:
                for other in self.grid[cell]:
                    # Skip if entity is already in set
                    if other in nearby_entities or other is entity:
                        continue
                        
                    # Use squared distance for faster comparison (avoid sqrt)
                    dx = other.x - ex
                    dy = other.y - ey
                    dist_squared = dx*dx + dy*dy
                    
                    # Check if within radius
                    if dist_squared <= radius_squared:
                        nearby_entities.add(other)
        
        # Convert to list for return
        return list(nearby_entities) 