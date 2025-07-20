import networkx as nx
import numpy as np
import random

class NetworkGenerator:
    """Generate realistic network topologies and traffic patterns"""
    
    @staticmethod
    def create_topology(n_nodes, topology_type='waxman'):
        """
        Create different network topologies
        Types: 'waxman', 'ba' (Barabási-Albert), 'er' (Erdős-Rényi), 'ws' (Watts-Strogatz)
        """
        if topology_type == 'waxman':
            # Waxman random graph - more realistic for geographic networks
            # Increase parameters for denser networks
            G = nx.waxman_graph(n_nodes, beta=0.6, alpha=0.2)
        elif topology_type == 'ba':
            # Scale-free network - increase connections
            m = max(3, n_nodes // 5)
            G = nx.barabasi_albert_graph(n_nodes, m)
        elif topology_type == 'er':
            # Random graph - increase edge probability
            p = 0.3 if n_nodes < 20 else 0.2
            G = nx.erdos_renyi_graph(n_nodes, p)
        else:  # ws
            # Small-world network - increase connections
            k = min(6, n_nodes - 1)
            G = nx.watts_strogatz_graph(n_nodes, k, 0.3)
        
        # Ensure connectivity
        if not nx.is_connected(G):
            # Add edges to largest component to make connected
            components = list(nx.connected_components(G))
            for i in range(1, len(components)):
                u = list(components[0])[0]
                v = list(components[i])[0]
                G.add_edge(u, v)
        
        # Add more edges if too sparse
        min_edges = n_nodes * 1.5
        while G.number_of_edges() < min_edges:
            # Add random edge
            u, v = random.sample(list(G.nodes()), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
        
        # Relabel nodes
        mapping = {i: f"n{i}" for i in range(n_nodes)}
        G = nx.relabel_nodes(G, mapping)
        
        return G
    
    @staticmethod
    def assign_capacities(G, capacity_distribution='realistic'):
        """
        Assign link capacities based on different distributions
        """
        capacities = {}
        
        if capacity_distribution == 'realistic':
            # Mix of capacities as in real networks
            # 40% at 1Gbps, 30% at 10Gbps, 20% at 40Gbps, 10% at 100Gbps
            capacity_choices = [1000, 10000, 40000, 100000]
            weights = [0.4, 0.3, 0.2, 0.1]
            
            for u, v in G.edges():
                capacity = random.choices(capacity_choices, weights=weights)[0]
                capacities[(u, v)] = capacity
                capacities[(v, u)] = capacity
                
        elif capacity_distribution == 'uniform':
            # All links same capacity
            for u, v in G.edges():
                capacities[(u, v)] = 10000
                capacities[(v, u)] = 10000
                
        else:  
            # Core links have higher capacity
            # Identify core nodes (high degree)
            degrees = dict(G.degree())
            avg_degree = np.mean(list(degrees.values()))
            
            for u, v in G.edges():
                if degrees[u] > avg_degree and degrees[v] > avg_degree:
                    capacity = 100000  # Core link
                elif degrees[u] > avg_degree or degrees[v] > avg_degree:
                    capacity = 40000   # Edge to core
                else:
                    capacity = 10000   # Edge link
                    
                capacities[(u, v)] = capacity
                capacities[(v, u)] = capacity
        
        return capacities
    
    @staticmethod
    def generate_traffic_matrix(G, pattern='gravity', intensity=0.5):
        """
        Generate traffic demands based on different patterns
        """
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        demands = {}
        
        if pattern == 'gravity':
            # Gravity model: traffic proportional to node degrees
            degrees = dict(G.degree())
            total_degree = sum(degrees.values())
            
            # Generate 40-60% of all possible flows
            n_flows = int(random.uniform(0.4, 0.6) * n_nodes * (n_nodes - 1))
            
            for _ in range(n_flows):
                # Select source and destination based on degree
                s = random.choices(nodes, weights=[degrees[n] for n in nodes])[0]
                t = random.choices(nodes, weights=[degrees[n] for n in nodes])[0]
                
                if s != t and (s, t) not in demands:
                    # Demand proportional to product of degrees
                    base_demand = (degrees[s] * degrees[t]) / (total_degree / n_nodes) ** 2
                    demand = base_demand * random.uniform(50, 500) * intensity
                    demands[(s, t)] = min(demand, 2000)  # Cap at 2Gbps
                    
        elif pattern == 'uniform':
            # Uniform random demands
            n_flows = int(0.5 * n_nodes * (n_nodes - 1))
            pairs = [(s, t) for s in nodes for t in nodes if s != t]
            selected_pairs = random.sample(pairs, n_flows)
            
            for s, t in selected_pairs:
                demands[(s, t)] = random.uniform(100, 1000) * intensity
                
        else:  # 'hotspot'
            # Some nodes generate/attract more traffic
            hotspots = random.sample(nodes, max(2, n_nodes // 5))
            
            for s in nodes:
                for t in nodes:
                    if s != t and random.random() < 0.5:
                        if s in hotspots or t in hotspots:
                            demands[(s, t)] = random.uniform(200, 1500) * intensity
                        else:
                            demands[(s, t)] = random.uniform(50, 500) * intensity
        
        return demands