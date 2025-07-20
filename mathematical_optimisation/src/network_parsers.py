import networkx as nx
import xml.etree.ElementTree as ET
import os

# Mapping country codes to full names for Geant
COUNTRY_NAMES = {
    'at': 'Austria',
    'be': 'Belgium',
    'ch': 'Switzerland',
    'cz': 'Czech Rep.',
    'de': 'Germany',
    'es': 'Spain',
    'fr': 'France',
    'gr': 'Greece',
    'hr': 'Croatia',
    'hu': 'Hungary',
    'ie': 'Ireland',
    'il': 'Israel',
    'it': 'Italy',
    'lu': 'Luxembourg',
    'nl': 'Netherlands',
    'ny': 'New York',
    'pl': 'Poland',
    'pt': 'Portugal',
    'se': 'Sweden',
    'si': 'Slovenia',
    'sk': 'Slovakia',
    'uk': 'UK'
}

def parse_atlanta_xml(xml_file_path):
    """Parse Atlanta network XML file from ZIB SNDlib"""
    try:
        if not os.path.exists(xml_file_path):
            print(f"Error: File {xml_file_path} not found")
            return None, None, None
            
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        print(f"Successfully loaded XML from {xml_file_path}")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Initialize capacities dictionary before usage
        capacities = {}
        
        # Parse nodes
        nodes_found = 0
        for node in root.iter():
            if 'node' in node.tag and node.get('id'):
                node_id = node.get('id')
                G.add_node(node_id)
                nodes_found += 1
        
        print(f"Found {nodes_found} nodes")
        
        # Parse links and capacities
        links_found = 0
        
        for link in root.iter():
            if 'link' in link.tag and link.get('id'):
                source = None
                target = None
                capacity = 0.0
                
                for elem in link.iter():
                    if 'source' in elem.tag and elem.text:
                        source = elem.text
                    elif 'target' in elem.tag and elem.text:
                        target = elem.text
                    elif 'capacity' in elem.tag and elem.text:
                        capacity = float(elem.text)
                
                if source and target:
                    G.add_edge(source, target)
                    capacities[(source, target)] = capacity
                    capacities[(target, source)] = capacity
                    links_found += 1
        
        print(f"Found {links_found} links")
        
        # Parse demands
        demands = {}
        demands_found = 0
        
        for demand in root.iter():
            if 'demand' in demand.tag and demand.get('id'):
                source = None
                target = None
                value = 0.0
                
                for elem in demand.iter():
                    if 'source' in elem.tag and elem.text:
                        source = elem.text
                    elif 'target' in elem.tag and elem.text:
                        target = elem.text
                    elif 'demandValue' in elem.tag and elem.text:
                        value = float(elem.text)
                
                if source and target and value > 0:
                    demands[(source, target)] = value
                    demands_found += 1
        
        print(f"Found {demands_found} demands")
        
        # Scale demands to make problem feasible
        total_demand = sum(demands.values())
        total_capacity = sum([c for c in capacities.values() if c > 0])/2
        
        print(f"\nAtlanta Network Summary:")
        print(f"  Nodes: {len(G.nodes())}")
        print(f"  Links: {len(G.edges())}")
        print(f"  Commodities: {len(demands)}")
        print(f"  Total demand: {total_demand:.0f} Mbps")
        print(f"  Total capacity: {total_capacity:.0f} Mbps")
        
        # Scale demands to 20% of total capacity for feasibility
        scale_factor = (total_capacity * 0.2) / total_demand if total_demand > 0 else 1.0
        scaled_demands = {k: v * scale_factor for k, v in demands.items()}
        
        print(f"\nScaling demands by factor {scale_factor:.3f} for feasibility")
        print(f"  Scaled total demand: {sum(scaled_demands.values()):.0f} Mbps")
        
        return G, capacities, scaled_demands
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
def parse_geant_xml(xml_file_path):
    """Parse Geant network XML file from ZIB SNDlib"""
    try:
        if not os.path.exists(xml_file_path):
            print(f"Error: File {xml_file_path} not found")
            return None, None, None
            
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        print(f"Successfully loaded Geant XML from {xml_file_path}")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Store coordinates
        coordinates = {}
        
        # Parse nodes with coordinates
        nodes_found = 0
        for node in root.iter():
            if 'node' in node.tag and node.get('id'):
                node_id = node.get('id')
                
                # Extract country code and create proper name
                country_code = node_id.split('.')[0][:2]  # Get first 2 chars before dot
                country_name = COUNTRY_NAMES.get(country_code, node_id)
                
                # Add node with country name as label
                G.add_node(node_id, label=country_name)
                
                # Extract coordinates
                x = None
                y = None
                for coord in node.iter():
                    if 'x' in coord.tag and coord.text:
                        x = float(coord.text)
                    elif 'y' in coord.tag and coord.text:
                        y = float(coord.text)
                
                if x is not None and y is not None:
                    coordinates[node_id] = (x, y)
                
                nodes_found += 1
        
        print(f"Found {nodes_found} nodes with proper country names")
        
        # Parse links and capacities 
        capacities = {}
        links_found = 0
        processed_links = set()
        
        for link in root.iter():
            if 'link' in link.tag and link.get('id'):
                source = None
                target = None
                capacity = 0.0
                
                for elem in link.iter():
                    if 'source' in elem.tag and elem.text:
                        source = elem.text
                    elif 'target' in elem.tag and elem.text:
                        target = elem.text
                    elif 'capacity' in elem.tag and elem.text:
                        capacity = float(elem.text)
                
                if source and target:
                    # Create canonical link representation
                    link_key = tuple(sorted([source, target]))
                    
                    # Only add edge if I haven't seen this link before
                    if link_key not in processed_links:
                        G.add_edge(source, target)
                        capacities[(source, target)] = capacity
                        capacities[(target, source)] = capacity
                        links_found += 1
                        processed_links.add(link_key)
        
        print(f"Found {links_found} unique links")
        
        # Parse demands
        demands = {}
        demands_found = 0
        
        for demand in root.iter():
            if 'demand' in demand.tag and demand.get('id'):
                source = None
                target = None
                value = 0.0
                
                for elem in demand.iter():
                    if 'source' in elem.tag and elem.text:
                        source = elem.text
                    elif 'target' in elem.tag and elem.text:
                        target = elem.text
                    elif 'demandValue' in elem.tag and elem.text:
                        value = float(elem.text)
                
                if source and target and value > 0:
                    demands[(source, target)] = value
                    demands_found += 1
        
        print(f"Found {demands_found} demands")
        
        # Summary
        total_demand = sum(demands.values())
        total_capacity = sum(capacities.values()) / 2
        
        print(f"\nGeant Network Summary:")
        print(f"  Nodes: {len(G.nodes())}")
        print(f"  Links: {len(G.edges())}")
        print(f"  Commodities: {len(demands)}")
        print(f"  Total demand: {total_demand:.0f} Mbps")
        print(f"  Total capacity: {total_capacity:.0f} Mbps")
        
        # Scale demands if needed
        scale_factor = 0.05  # Conservative scaling for Geant
        scaled_demands = {k: v * scale_factor for k, v in demands.items()}
        
        print(f"\nScaling demands by factor {scale_factor:.3f} for feasibility")
        print(f"  Scaled total demand: {sum(scaled_demands.values()):.0f} Mbps")
        
        return G, capacities, scaled_demands
        
    except Exception as e:
        print(f"Error parsing Geant XML: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def compute_ospf_routing(G, demands):
    """Compute OSPF routing matrix using shortest paths"""
    routing_matrix = {}
    
    for (s, t), demand in demands.items():
        try:
            path = nx.shortest_path(G, s, t)
            for i in range(len(path) - 1):
                arc = (path[i], path[i+1])
                routing_matrix[(s,t), arc] = 1.0
        except nx.NetworkXNoPath:
            pass
    
    return routing_matrix

def get_node_labels(G):
    """Get appropriate node labels based on network type"""
    # Check if nodes have 'label' attribute (for Geant)
    labels = nx.get_node_attributes(G, 'label')
    if labels:
        return labels
    
    # Otherwise, return node IDs as labels (for Atlanta)
    return {node: node for node in G.nodes()}