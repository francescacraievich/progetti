import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igp_optimizer import IGPWeightOptimizer
from ecmp_handler import ECMPHandler
from combined_routing_optimizer import CombinedRoutingOptimizer

class IterativeOptimizer:
    """
    Implement iterative process from Section 4.1 of the paper
    """
    def __init__(self, G, capacities, demands, max_iterations=5):
        self.G = G
        self.capacities = capacities
        self.demands = demands
        self.max_iterations = max_iterations
    
    def _calculate_mpls_flow_for_demand(self, s, t, w_values):
        """Calculate MPLS flow for a specific demand"""
        total_flow = 0
        
        # Find all MPLS flows from source s
        for (v, arc), flow in w_values.items():
            if v == s and flow > 0.01:
                # Check if this flow contributes to demand (s,t)
                # This is a simplified approach - in reality would need path tracking
                if arc[1] == t or self._is_on_path_to(arc[1], t, w_values, v):
                    total_flow += flow
        
        return total_flow
    
    def _is_on_path_to(self, node, destination, w_values, origin):
        """Check if node is on a path to destination in MPLS flows"""
        # Simplified check - in practice would need full path reconstruction
        for (v, arc), flow in w_values.items():
            if v == origin and arc[0] == node and flow > 0.01:
                if arc[1] == destination:
                    return True
        return False
        
    def optimize_iteratively(self, initial_weights=None, delta=0, verbose=True):
        """
        Iterative optimization process:
        1. Optimize IGP weights
        2. Find complementary LSPs
        3. Fix MPLS flows and re-optimize weights
        """
        if verbose:
            print("\n" + "="*60)
            print("STARTING ITERATIVE OPTIMIZATION PROCESS")
            print("="*60)
        
        # Initialize
        if initial_weights is None:
            igp_optimizer = IGPWeightOptimizer(self.G, self.capacities, self.demands)
            current_weights = igp_optimizer.optimize()
        else:
            current_weights = initial_weights
        
        best_result = None
        best_utilization = float('inf')
        best_weights = current_weights.copy()
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Step 1: Compute routing matrix with current weights (with ECMP)
            if iteration == 0:
                routing_matrix = ECMPHandler.compute_ecmp_routing(
                    self.G, self.demands, current_weights
                )
                current_demands = self.demands
            else:
                # Use modified demands after fixing MPLS flows
                routing_matrix = ECMPHandler.compute_ecmp_routing(
                    self.G, modified_demands, current_weights
                )
                current_demands = modified_demands
            
            # Step 2: Solve combined routing problem with ECMP
            optimizer = CombinedRoutingOptimizer(
                self.G, self.capacities, current_demands, routing_matrix, use_ecmp=True
            )
            
            result = optimizer.solve_nominal(delta=delta, verbose=False)
            
            if result['status'] != 'optimal':
                if verbose:
                    print("  No optimal solution found!")
                break
            
            if verbose:
                print(f"  Combined utilization: {result['u_max_percent']:.1f}%")
                print(f"  Number of LSPs: {result['lsp_count']}")
            
            # Check for improvement
            if result['u_max_percent'] >= best_utilization - 0.1:  # No significant improvement
                if verbose:
                    print("  No significant improvement, stopping iterations")
                break
            
            best_result = result
            best_utilization = result['u_max_percent']
            best_weights = current_weights.copy()
            
            # Step 3: Fix MPLS flows and create modified demands
            modified_demands = {}
            modified_capacities = self.capacities.copy()
            
            # Calculate reduced demands based on MPLS flows
            for (s, t), demand in self.demands.items():
                mpls_flow = self._calculate_mpls_flow_for_demand(s, t, result['w_values'])
                modified_demands[(s, t)] = max(0, demand - mpls_flow)
            
            # Update capacities by subtracting MPLS flows
            for (v, arc), flow in result['w_values'].items():
                if arc in modified_capacities:
                    modified_capacities[arc] = max(0, modified_capacities[arc] - flow)
            
            # Re-optimize weights with modified network
            if iteration < self.max_iterations - 1:
                if verbose:
                    print("  Re-optimizing IGP weights with fixed MPLS flows...")
                igp_optimizer = IGPWeightOptimizer(
                    self.G, modified_capacities, modified_demands
                )
                current_weights = igp_optimizer.optimize()
        
        if verbose:
            print(f"\nIterative optimization complete!")
            print(f"Best utilization achieved: {best_utilization:.1f}%")
        
        return best_result, best_weights