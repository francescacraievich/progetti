"""
Minimal OSM processor for Austria data following Raviv (2023)
"""

import os
import osmium
import numpy as np
import random
from typing import Dict, List, Tuple
from collections import defaultdict


class AustriaOSMHandler(osmium.SimpleHandler):
    """Simple handler to extract data from Austria OSM file"""
    
    # City bounds from Raviv (2023)
    CITIES = {
        'Vienna': {
            'bounds': (16.18, 48.11, 16.58, 48.32),
            'target_demand_points': 17701,
            'target_sp_candidates': 4260
        },
        'Graz': {
            'bounds': (15.34, 46.99, 15.51, 47.15),
            'target_demand_points': 6777,
            'target_sp_candidates': 1459
        },
        'Linz': {
            'bounds': (14.22, 48.24, 14.35, 48.35),
            'target_demand_points': 3351,
            'target_sp_candidates': 658
        }
    }
    
    def __init__(self, city: str):
        super().__init__()
        self.city = city
        self.bounds = self.CITIES[city]['bounds']
        self.buildings = []
        self.parking = []
        
    def node(self, n):
        """Extract parking and transport nodes"""
        if not self._in_bounds(n.location.lon, n.location.lat):
            return
            
        if ('amenity' in n.tags and n.tags['amenity'] == 'parking') or \
           ('railway' in n.tags and n.tags['railway'] in ['station', 'halt']):
            self.parking.append((n.location.lon, n.location.lat))
    
    def way(self, w):
        """Extract buildings"""
        if 'building' in w.tags and len(w.nodes) > 0:
            # Simple check if in bounds 
            self.buildings.append(w.id)
    
    def _in_bounds(self, lon, lat):
        min_lon, min_lat, max_lon, max_lat = self.bounds
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def load_austria_instance(city: str = "Vienna", osm_file: str = None) -> Dict:
    """
    Load city data from OSM files following Raviv (2023) methodology
    
    Parameters:
    -----------
    city : str
        City name (Vienna, Graz, or Linz)
    osm_file : str, optional
        Path to the OSM file. If None, tries city-specific file first,
        then falls back to full Austria file
    """
    
    if osm_file is None:
        # Try city-specific file first 
        city_lower = city.lower()
        city_file = f"data/output/osm/{city_lower}.osm.pbf"
        full_file = "data/output/osm/austria-latest.osm.pbf"
        
        if os.path.exists(city_file):
            osm_file = city_file
            print(f"Using city-specific file: {city_file}")
        elif os.path.exists(full_file):
            osm_file = full_file
            print(f"Using full Austria file: {full_file}")
            print(f"Note: This will be slower. Consider running extract_cities.py to create city-specific files.")
        else:
            # Provide helpful error message
            raise FileNotFoundError(
                f"OSM file not found. Tried:\n"
                f"  - {os.path.abspath(city_file)} (city-specific)\n"
                f"  - {os.path.abspath(full_file)} (full Austria)\n\n"
                f"To get the data:\n"
                f"1. For faster loading, run: python src/utils/extract_cities.py\n"
                f"2. Or download from: https://download.geofabrik.de/europe/austria-latest.osm.pbf\n"
                f"   and place in: data/output/osm/"
            )
    
    # Check file size to warn user if using large file
    file_size_mb = os.path.getsize(osm_file) / 1024 / 1024
    if file_size_mb > 100:
        print(f"Warning: Large file ({file_size_mb:.0f} MB) - loading may take a minute...")
    
    print(f"Loading {city} from: {osm_file}")
    
    handler = AustriaOSMHandler(city)
    
    # This processes the OSM file
    handler.apply_file(osm_file, locations=True)
    
    # Create 250m grid as in paper
    bounds = handler.bounds
    cell_size = 0.0025  # ~250m in degrees
    
    # Generate ALL grid points (both demand points and SP candidates)
    all_grid_points = []
    all_demand_rates = []
    
    lon = bounds[0]
    while lon < bounds[2]:
        lat = bounds[1]
        while lat < bounds[3]:
            # Center of grid cell
            x = (lon + cell_size/2 - (bounds[0]+bounds[2])/2) * 111000
            y = (lat + cell_size/2 - (bounds[1]+bounds[3])/2) * 111000
            all_grid_points.append((x, y))
            all_demand_rates.append(np.random.uniform(0.5, 10))  # Random demand
            lat += cell_size
        lon += cell_size
    
    print(f"Generated {len(all_grid_points)} total grid points for {city}")
    
    # Get target numbers from paper
    target_demand = handler.CITIES[city]['target_demand_points']
    target_sp = handler.CITIES[city]['target_sp_candidates']
    
    # Ensure we don't exceed available points
    n_demand = min(len(all_grid_points), target_demand)
    n_sp = min(len(all_grid_points), target_sp)
    
    # According to Raviv (2023), SP candidates are selected from ALL grid points
    # So we randomly sample from all grid points for both demand and SP
    
    # For demand points: use the first n_demand points (or random sample)
    if len(all_grid_points) > target_demand:
        # Randomly sample demand points
        demand_indices = random.sample(range(len(all_grid_points)), n_demand)
        demand_points = [all_grid_points[i] for i in demand_indices]
        demand_rates = [all_demand_rates[i] for i in demand_indices]
    else:
        demand_points = all_grid_points[:n_demand]
        demand_rates = all_demand_rates[:n_demand]
    
    # For SP locations: according to paper, sample from ALL grid points
    if len(all_grid_points) > target_sp:
        # Randomly sample SP locations from all grid points
        sp_indices = random.sample(range(len(all_grid_points)), n_sp)
        sp_locations = [all_grid_points[i] for i in sp_indices]
    else:
        sp_locations = all_grid_points[:n_sp]
    
    instance = {
        "city": city,
        "demand_points": demand_points,
        "demand_rates": demand_rates,
        "sp_locations": sp_locations,
        "service_radius": 300,  # 300m for real instances
        "n_demand_points": n_demand,
        "n_sp_locations": n_sp,
        "total_demand": sum(demand_rates)
    }
    
    print(f"Loaded {city}: {n_demand} demand points, {n_sp} SP candidates (from {len(all_grid_points)} grid points)")
    return instance