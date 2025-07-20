import networkx as nx
import random
from collections import defaultdict

class IGPWeightOptimizer:
    """
    Improved Tabu Search implementation for IGP weight optimization
    Based on references from the paper with enhanced features
    """
    def __init__(self, G, capacities, demands, tabu_size=20, max_iterations=100):
        self.G = G
        self.capacities = capacities
        self.demands = demands
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.tabu_list = []
        
        # Enhanced features
        self.aspiration_criteria = True
        self.diversification_threshold = 20
        self.frequency_memory = defaultdict(int)
        self.iteration_without_improvement = 0
        self.best_known_value = float('inf')
        
    def evaluate_weights(self, weights):
        """Evaluate max utilization for given weights"""
        # Compute shortest paths with given weights
        G_weighted = self.G.copy()
        for u, v in G_weighted.edges():
            G_weighted[u][v]['weight'] = weights.get((u, v), 1)
        
        # Compute routing matrix
        routing_matrix = {}
        for (s, t), demand in self.demands.items():
            try:
                path = nx.shortest_path(G_weighted, s, t, weight='weight')
                for i in range(len(path) - 1):
                    arc = (path[i], path[i+1])
                    routing_matrix[(s, t), arc] = 1.0
            except nx.NetworkXNoPath:
                pass
        
        # Calculate max utilization
        arc_loads = defaultdict(float)
        for (commodity, arc), value in routing_matrix.items():
            if commodity in self.demands:
                arc_loads[arc] += self.demands[commodity] * value
        
        max_util = 0
        for arc, load in arc_loads.items():
            if arc in self.capacities and self.capacities[arc] > 0:
                util = load / self.capacities[arc]
                max_util = max(max_util, util)
        
        return max_util
    
    def is_tabu(self, solution, current_value):
        """Check tabu status with aspiration criteria"""
        solution_hash = hash(frozenset(solution.items()))
        
        if solution_hash in self.tabu_list:
            # Aspiration criteria: accept if better than best known
            if self.aspiration_criteria and current_value < self.best_known_value:
                return False
            return True
        return False
    
    def generate_neighbor(self, weights):
        """Generate neighbor solution by modifying weights"""
        new_weights = weights.copy()
        edges = list(self.G.edges())
        
        # Use frequency memory for diversification
        if self.iteration_without_improvement > self.diversification_threshold:
            # Diversification: modify edges that have been changed less frequently
            sorted_edges = sorted(edges, key=lambda e: self.frequency_memory[e])
            edges_to_modify = sorted_edges[:min(5, len(sorted_edges))]
        else:
            # Normal operation: random selection
            num_changes = random.randint(1, min(3, len(edges)))
            edges_to_modify = random.sample(edges, num_changes)
        
        for edge in edges_to_modify:
            # Change weight within reasonable range [1, 20]
            new_weights[edge] = random.randint(1, 20)
            new_weights[(edge[1], edge[0])] = new_weights[edge]
            self.frequency_memory[edge] += 1
        
        return new_weights
    
    def diversify(self):
        """Diversification strategy when stuck"""
        print("  Applying diversification strategy...")
        self.iteration_without_improvement = 0
        # Clear part of tabu list to allow revisiting
        self.tabu_list = self.tabu_list[-self.tabu_size//2:]
    
    def optimize(self):
        """Run improved Tabu Search optimization"""
        print("Starting IGP weight optimization with Enhanced Tabu Search...")
        
        # Initialize with unit weights
        current_weights = {(u, v): 1 for u, v in self.G.edges()}
        current_weights.update({(v, u): 1 for u, v in self.G.edges()})
        
        best_weights = current_weights.copy()
        best_util = self.evaluate_weights(best_weights)
        self.best_known_value = best_util
        
        print(f"Initial utilization: {best_util*100:.1f}%")
        
        for iteration in range(self.max_iterations):
            # Generate neighbors
            neighbors = []
            attempts = 0
            max_attempts = 30
            
            while len(neighbors) < 10 and attempts < max_attempts:
                neighbor = self.generate_neighbor(current_weights)
                neighbor_hash = hash(frozenset(neighbor.items()))
                util = self.evaluate_weights(neighbor)
                
                if not self.is_tabu(neighbor, util):
                    neighbors.append((neighbor, util, neighbor_hash))
                attempts += 1
            
            if not neighbors:
                # If no non-tabu neighbors found, apply diversification
                self.diversify()
                continue
            
            # Select best neighbor
            neighbors.sort(key=lambda x: x[1])
            current_weights, current_util, current_hash = neighbors[0]
            
            # Update tabu list
            self.tabu_list.append(current_hash)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)
            
            # Update best solution
            if current_util < best_util:
                best_weights = current_weights.copy()
                best_util = current_util
                self.best_known_value = best_util
                self.iteration_without_improvement = 0
                print(f"  Iteration {iteration}: New best utilization = {best_util*100:.1f}%")
            else:
                self.iteration_without_improvement += 1
            
            # Check for diversification need
            if self.iteration_without_improvement > self.diversification_threshold:
                self.diversify()
        
        print(f"Optimization complete. Best utilization: {best_util*100:.1f}%")
        return best_weights