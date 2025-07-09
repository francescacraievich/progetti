#!/usr/bin/env python3
"""
Scalability testing for Combined IGP/MPLS-TE Routing Optimization
Tests performance with increasing network complexity
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from test import CombinedRoutingOptimizer, compute_ospf_routing
import random

class ScalabilityTester:
    def __init__(self):
        self.results = {
            'nodes': [],
            'edges': [],
            'commodities': [],
            'variables': [],
            'constraints': [],
            'solve_time': [],
            'u_max': [],
            'lsp_count': []
        }
    
    def generate_network(self, n_nodes, edge_probability=0.3):
        """Generate a random connected network"""
        # Use Erdős-Rényi model and ensure connectivity
        while True:
            G = nx.erdos_renyi_graph(n_nodes, edge_probability)
            if nx.is_connected(G):
                break
        
        # Relabel nodes
        mapping = {i: f"n{i}" for i in range(n_nodes)}
        G = nx.relabel_nodes(G, mapping)
        
        # Assign capacities (random between 1000-10000 Mbps)
        capacities = {}
        for i, j in G.edges():
            capacity = random.randint(1000, 10000)
            capacities[(i,j)] = capacity
            capacities[(j,i)] = capacity
        
        return G, capacities
    
    def generate_demands(self, G, n_commodities=None):
        """Generate random traffic demands"""
        nodes = list(G.nodes())
        
        if n_commodities is None:
            # Default: 30% of all possible O-D pairs
            n_commodities = int(0.3 * len(nodes) * (len(nodes) - 1))
        
        demands = {}
        commodity_pairs = set()
        
        while len(commodity_pairs) < n_commodities:
            s = random.choice(nodes)
            t = random.choice(nodes)
            if s != t:
                commodity_pairs.add((s, t))
        
        for s, t in commodity_pairs:
            # Random demand between 10-500 Mbps
            demands[(s,t)] = random.randint(10, 500)
        
        return demands
    
    def test_instance(self, n_nodes, edge_prob=0.3, commodity_ratio=0.3):
        """Test a single instance and collect metrics"""
        print(f"\nTesting network with {n_nodes} nodes...")
        
        # Generate network
        G, capacities = self.generate_network(n_nodes, edge_prob)
        n_edges = len(G.edges())
        
        # Generate demands
        n_commodities = int(commodity_ratio * n_nodes * (n_nodes - 1))
        demands = self.generate_demands(G, n_commodities)
        
        # Compute OSPF routing
        routing_matrix = compute_ospf_routing(G, demands)
        
        # Create optimizer
        optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
        
        # Count variables and constraints
        n_arcs = 2 * n_edges
        n_origin_nodes = len(optimizer.origin_nodes)
        n_vars = 1 + len(demands) + n_origin_nodes * n_arcs  # u_max + is_vars + w_vars
        n_constraints = n_arcs + n_origin_nodes * n_nodes  # utilization + flow conservation
        
        print(f"  Edges: {n_edges}, Commodities: {len(demands)}")
        print(f"  Variables: {n_vars}, Constraints: {n_constraints}")
        
        # Solve
        start_time = time.time()
        model, u_max, w_values, is_values = optimizer.solve_nominal(delta=0.0001)
        solve_time = time.time() - start_time
        
        if model and model.status == GRB.OPTIMAL:
            # Count LSPs
            lsp_count = sum(1 for v,arc in w_values.keys() 
                           if arc[0] == v and w_values[v,arc] > 0.01)
            
            print(f"  Solve time: {solve_time:.2f}s")
            print(f"  Max utilization: {u_max:.1f}%")
            print(f"  LSP count: {lsp_count}")
            
            # Store results
            self.results['nodes'].append(n_nodes)
            self.results['edges'].append(n_edges)
            self.results['commodities'].append(len(demands))
            self.results['variables'].append(n_vars)
            self.results['constraints'].append(n_constraints)
            self.results['solve_time'].append(solve_time)
            self.results['u_max'].append(u_max)
            self.results['lsp_count'].append(lsp_count)
            
            return True
        else:
            print("  Failed to find optimal solution!")
            return False
    
    def run_tests(self, node_sizes=[8, 12, 16, 20, 25, 30, 35, 40]):
        """Run scalability tests on networks of increasing size"""
        print("=== Scalability Testing for Combined Routing ===")
        
        for n_nodes in node_sizes:
            success = self.test_instance(n_nodes)
            if not success:
                print(f"Stopping tests - failed at {n_nodes} nodes")
                break
        
        self.plot_results()
    
    def plot_results(self):
        """Plot scalability results"""
        if not self.results['nodes']:
            print("No results to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Scalability Analysis of Combined IGP/MPLS-TE Routing', fontsize=16)
        
        # Plot 1: Solve time vs network size
        ax1 = axes[0, 0]
        ax1.plot(self.results['nodes'], self.results['solve_time'], 'b-o')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Solve Time (seconds)')
        ax1.set_title('Computational Time Scaling')
        ax1.grid(True)
        
        # Plot 2: Problem size
        ax2 = axes[0, 1]
        ax2.plot(self.results['nodes'], self.results['variables'], 'r-s', label='Variables')
        ax2.plot(self.results['nodes'], self.results['constraints'], 'g-^', label='Constraints')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Count')
        ax2.set_title('Problem Size Growth')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Solution quality
        ax3 = axes[1, 0]
        ax3.plot(self.results['nodes'], self.results['u_max'], 'purple', marker='o')
        ax3.set_xlabel('Number of Nodes')
        ax3.set_ylabel('Max Link Utilization (%)')
        ax3.set_title('Solution Quality')
        ax3.grid(True)
        
        # Plot 4: LSP count
        ax4 = axes[1, 1]
        ax4.plot(self.results['nodes'], self.results['lsp_count'], 'orange', marker='d')
        ax4.set_xlabel('Number of Nodes')
        ax4.set_ylabel('Number of LSPs')
        ax4.set_title('LSP Complexity')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('scalability_results.png', dpi=150)
        plt.show()
        
        # Print summary statistics
        print("\n=== Scalability Summary ===")
        print(f"Tested networks from {min(self.results['nodes'])} to {max(self.results['nodes'])} nodes")
        print(f"Maximum solve time: {max(self.results['solve_time']):.2f} seconds")
        print(f"Average max utilization: {np.mean(self.results['u_max']):.1f}%")
        
        # Analyze computational complexity
        if len(self.results['nodes']) > 3:
            # Fit polynomial to solve time
            coeffs = np.polyfit(self.results['nodes'], self.results['solve_time'], 2)
            print(f"\nSolve time complexity: O(n^{np.log(coeffs[0])/np.log(10):.2f})")

class SurvivabilityTester(ScalabilityTester):
    """Extended tester for survivability constraints"""
    
    def test_survivable_instance(self, n_nodes, edge_prob=0.3, commodity_ratio=0.3):
        """Test with survivability constraints (simplified version)"""
        print(f"\nTesting survivable routing with {n_nodes} nodes...")
        
        # Generate network
        G, capacities = self.generate_network(n_nodes, edge_prob)
        n_edges = len(G.edges())
        
        # For scalability testing, only consider subset of failure scenarios
        max_failures = min(5, n_edges)  # Limit failures for larger networks
        
        # Generate demands
        n_commodities = int(commodity_ratio * n_nodes * (n_nodes - 1))
        demands = self.generate_demands(G, n_commodities)
        
        print(f"  Testing with {max_failures} failure scenarios")
        
        # This is a simplified test - full implementation would require
        # failure-specific routing matrices as in the paper
        
        return True

def main():
    # Test 1: Basic scalability
    print("Running basic scalability tests...")
    tester = ScalabilityTester()
    tester.run_tests(node_sizes=[8, 10, 12, 15, 18, 20, 25, 30])
    
    # Test 2: Effect of network density
    print("\n\nTesting effect of network density...")
    density_results = {}
    
    for density in [0.2, 0.3, 0.4, 0.5]:
        print(f"\nTesting with edge probability {density}")
        tester = ScalabilityTester()
        tester.test_instance(20, edge_prob=density)
        if tester.results['solve_time']:
            density_results[density] = {
                'time': tester.results['solve_time'][0],
                'u_max': tester.results['u_max'][0],
                'lsps': tester.results['lsp_count'][0]
            }
    
    # Plot density results
    if density_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        densities = list(density_results.keys())
        times = [density_results[d]['time'] for d in densities]
        utilizations = [density_results[d]['u_max'] for d in densities]
        
        ax1.plot(densities, times, 'b-o')
        ax1.set_xlabel('Edge Probability')
        ax1.set_ylabel('Solve Time (s)')
        ax1.set_title('Network Density vs Solve Time')
        ax1.grid(True)
        
        ax2.plot(densities, utilizations, 'r-s')
        ax2.set_xlabel('Edge Probability')
        ax2.set_ylabel('Max Utilization (%)')
        ax2.set_title('Network Density vs Solution Quality')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('density_analysis.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    main()