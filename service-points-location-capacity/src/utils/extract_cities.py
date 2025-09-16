"""
Extract individual city data from Austria OSM file
"""
import os
import osmium


class CityExtractor(osmium.SimpleHandler):
    """Handler to extract OSM data within city boundaries"""
    
    def __init__(self, output_writer, bounds):
        super().__init__()
        self.writer = output_writer
        self.bounds = bounds
        
    def node(self, n):
        """Process nodes - keep if within bounds"""
        if self._in_bounds(n.location.lon, n.location.lat):
            self.writer.add_node(n)
    
    def way(self, w):
        """Process ways - keep if at least one node is within bounds"""
        for n in w.nodes:
            if n.location and self._in_bounds(n.location.lon, n.location.lat):
                self.writer.add_way(w)
                break
                
    def _in_bounds(self, lon, lat):
        """Check if coordinates are within city bounds"""
        return self.bounds[0] <= lon <= self.bounds[2] and self.bounds[1] <= lat <= self.bounds[3]


def extract_cities():
    """Extract Vienna, Graz, and Linz from the full Austria OSM file"""
    
    # City boundaries from Raviv (2023)
    cities = {
        'vienna': (16.18, 48.11, 16.58, 48.32),
        'graz': (15.34, 46.99, 15.51, 47.15),
        'linz': (14.22, 48.24, 14.35, 48.35)
    }
    
    # Input and output paths
    input_file = 'data/output/osm/austria-latest.osm.pbf'
    output_dir = 'data/output/osm'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print(f"Please ensure austria-latest.osm.pbf is in the correct location")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting extraction from {input_file}")
    print(f"File size: {os.path.getsize(input_file) / 1024 / 1024:.1f} MB")
    
    # Extract each city
    for city, bounds in cities.items():
        output_file = os.path.join(output_dir, f'{city}.osm.pbf')
        
        print(f"\nExtracting {city.capitalize()}...")
        print(f"  Bounds: {bounds}")
        print(f"  Output: {output_file}")
        
        try:
            writer = osmium.SimpleWriter(output_file)
            handler = CityExtractor(writer, bounds)
            handler.apply_file(input_file)
            writer.close()
            
            # Show extracted file size
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            print(f"  ✓ Extracted successfully ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"  ✗ Error extracting {city}: {e}")
    
    print("\nExtraction complete!")



if __name__ == "__main__":
    extract_cities()