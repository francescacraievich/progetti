from collections import defaultdict

class PathDecomposer:
    """
    Path decomposition algorithm to extract LSP paths from flow solution
    Reference: Ahuja et al. [49]
    """
    @staticmethod
    def decompose_flow(G, w_values, origin_nodes):
        """
        Decompose flow variables into explicit paths
        Returns list of (source, destination, path, flow_value)
        """
        paths = []
        
        for v in origin_nodes:
            # Build residual graph for origin v
            residual = defaultdict(float)
            for (origin, arc), flow in w_values.items():
                if origin == v and flow > 0.01:
                    residual[arc] = flow
            
            # Extract paths using flow decomposition
            while residual:
                # Find a path from v to any destination with positive flow
                path = PathDecomposer._find_path(v, residual)
                if not path:
                    break
                
                # Find minimum flow along path
                min_flow = float('inf')
                for i in range(len(path) - 1):
                    arc = (path[i], path[i+1])
                    if arc in residual:
                        min_flow = min(min_flow, residual[arc])
                
                if min_flow == float('inf') or min_flow <= 0.01:
                    break
                
                # Subtract flow along path
                for i in range(len(path) - 1):
                    arc = (path[i], path[i+1])
                    residual[arc] -= min_flow
                    if residual[arc] <= 0.01:
                        del residual[arc]
                
                # Record path
                paths.append((v, path[-1], path, min_flow))
        
        return paths
    
    @staticmethod
    def _find_path(source, residual):
        """Find a path from source using BFS"""
        visited = {source}
        queue = [(source, [source])]
        
        while queue:
            node, path = queue.pop(0)
            
            # Check outgoing arcs
            for arc, flow in residual.items():
                if arc[0] == node and arc[1] not in visited and flow > 0.01:
                    new_path = path + [arc[1]]
                    
                    # Check if a node is reached with no outgoing flow (destination)
                    has_outgoing = any(a[0] == arc[1] for a in residual if residual[a] > 0.01)
                    if not has_outgoing:
                        return new_path
                    
                    visited.add(arc[1])
                    queue.append((arc[1], new_path))
        
        return None