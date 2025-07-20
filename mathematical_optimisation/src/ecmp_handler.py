import networkx as nx

class ECMPHandler:
    """Handle Equal Cost Multi-Path routing"""
    @staticmethod
    def compute_ecmp_routing(G, demands, weights=None):
        """
        Compute ECMP routing matrix considering multiple shortest paths
        """
        if weights is None:
            weights = {(u, v): 1 for u, v in G.edges()}
            weights.update({(v, u): 1 for u, v in G.edges()})
        
        # Create weighted graph
        G_weighted = G.copy()
        for u, v in G.edges():
            G_weighted[u][v]['weight'] = weights.get((u, v), 1)
        
        routing_matrix = {}
        
        for (s, t), demand in demands.items():
            try:
                # Find all shortest paths
                paths = list(nx.all_shortest_paths(G_weighted, s, t, weight='weight'))
                
                if paths:
                    # Split traffic equally among paths
                    flow_per_path = 1.0 / len(paths)
                    
                    for path in paths:
                        for i in range(len(path) - 1):
                            arc = (path[i], path[i+1])
                            if ((s, t), arc) not in routing_matrix:
                                routing_matrix[(s, t), arc] = 0
                            routing_matrix[(s, t), arc] += flow_per_path
            except nx.NetworkXNoPath:
                pass
        
        return routing_matrix