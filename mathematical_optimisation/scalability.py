#!/usr/bin/env python3
"""
Improved Scalability Testing for Combined IGP/MPLS-TE Routing
Generates more realistic traffic patterns and network topologies
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
from test import CombinedRoutingOptimizer, compute_ospf_routing

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
                
        else:  # 'tiered'
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

class ScalabilityTester:
    def __init__(self):
        self.results = []
        self.generator = NetworkGenerator()
    
    def test_instance(self, n_nodes, topology='waxman', traffic_pattern='gravity', 
                     delta=1e-7, verbose=False):
        """Test a single instance"""
        
        # Generate network
        G = self.generator.create_topology(n_nodes, topology)
        capacities = self.generator.assign_capacities(G, 'realistic')
        
        # Generate traffic with appropriate intensity
        # Increase intensity to create more congestion
        intensity = 0.8 / np.sqrt(n_nodes / 15)
        demands = self.generator.generate_traffic_matrix(G, traffic_pattern, intensity)
        
        # Compute routing
        routing_matrix = compute_ospf_routing(G, demands)
        
        # Create optimizer
        optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
        
        # Compute IGP-only utilization
        igp_util, _, _ = optimizer.compute_igp_only_utilization()
        
        # Skip if IGP routing is not congested enough
        if igp_util < 50:  # Increased threshold
            if verbose:
                print(f"  Skipping - IGP util too low: {igp_util:.1f}%")
            return None
        
        # Solve combined routing with delta=0 for better MPLS usage
        start_time = time.time()
        result = optimizer.solve_nominal(delta=0, verbose=False)
        solve_time = time.time() - start_time
        
        if result['status'] == 'optimal':
            return {
                'nodes': n_nodes,
                'edges': G.number_of_edges(),
                'commodities': len(demands),
                'igp_util': igp_util,
                'combined_util': result['u_max_percent'],
                'improvement': (igp_util - result['u_max_percent']) / igp_util * 100,
                'lsp_count': result['lsp_count'],
                'igp_percent': result['igp_percent'],
                'mpls_percent': result['mpls_percent'],
                'solve_time': solve_time,
                'total_demand': sum(demands.values()),
                'total_capacity': sum(capacities.values()) / 2
            }
        else:
            return None
    
    def run_size_scaling_test(self, sizes=[8, 10, 12, 15, 20, 25, 30], 
                             runs_per_size=3):
        """Test scaling with network size"""
        print("=== Network Size Scaling Test ===")
        print(f"{'Nodes':>6} | {'Edges':>6} | {'Flows':>6} | "
              f"{'IGP%':>6} | {'Comb%':>6} | {'Impr%':>6} | "
              f"{'LSPs':>5} | {'Time(s)':>8}")
        print("-" * 70)
        
        results_by_size = {}
        
        for size in sizes:
            size_results = []
            attempts = 0
            
            while len(size_results) < runs_per_size and attempts < runs_per_size * 3:
                attempts += 1
                result = self.test_instance(size, topology='waxman', 
                                          traffic_pattern='gravity')
                if result:
                    size_results.append(result)
                    print(f"{result['nodes']:6d} | {result['edges']:6d} | "
                          f"{result['commodities']:6d} | {result['igp_util']:6.1f} | "
                          f"{result['combined_util']:6.1f} | {result['improvement']:6.1f} | "
                          f"{result['lsp_count']:5d} | {result['solve_time']:8.3f}")
            
            if size_results:
                # Average results for this size
                avg_result = {}
                for key in size_results[0].keys():
                    if isinstance(size_results[0][key], (int, float)):
                        avg_result[key] = np.mean([r[key] for r in size_results])
                    else:
                        avg_result[key] = size_results[0][key]
                
                results_by_size[size] = avg_result
                self.results.extend(size_results)
        
        return results_by_size
    
    def run_topology_comparison(self, n_nodes=20):
        """Compare different topology types"""
        print(f"\n=== Topology Comparison (n={n_nodes}) ===")
        topologies = ['waxman', 'ba', 'er', 'ws']
        
        print(f"{'Topology':>10} | {'Edges':>6} | {'IGP%':>6} | "
              f"{'Comb%':>6} | {'Impr%':>6} | {'LSPs':>5}")
        print("-" * 55)
        
        for topo in topologies:
            result = self.test_instance(n_nodes, topology=topo)
            if result:
                print(f"{topo:>10} | {result['edges']:6d} | "
                      f"{result['igp_util']:6.1f} | {result['combined_util']:6.1f} | "
                      f"{result['improvement']:6.1f} | {result['lsp_count']:5d}")
    
    def run_traffic_pattern_comparison(self, n_nodes=20):
        """Compare different traffic patterns"""
        print(f"\n=== Traffic Pattern Comparison (n={n_nodes}) ===")
        patterns = ['gravity', 'uniform', 'hotspot']
        
        print(f"{'Pattern':>10} | {'Flows':>6} | {'IGP%':>6} | "
              f"{'Comb%':>6} | {'Impr%':>6} | {'LSPs':>5}")
        print("-" * 55)
        
        for pattern in patterns:
            result = self.test_instance(n_nodes, traffic_pattern=pattern)
            if result:
                print(f"{pattern:>10} | {result['commodities']:6d} | "
                      f"{result['igp_util']:6.1f} | {result['combined_util']:6.1f} | "
                      f"{result['improvement']:6.1f} | {result['lsp_count']:5d}")
    
    def plot_results(self, results_by_size):
        """Create comprehensive plots of scalability results"""
        if not results_by_size:
            print("No results to plot!")
            return
        
        sizes = sorted(results_by_size.keys())
        
        # Extract data for plotting
        edges = [results_by_size[s]['edges'] for s in sizes]
        igp_utils = [results_by_size[s]['igp_util'] for s in sizes]
        combined_utils = [results_by_size[s]['combined_util'] for s in sizes]
        improvements = [results_by_size[s]['improvement'] for s in sizes]
        lsp_counts = [results_by_size[s]['lsp_count'] for s in sizes]
        solve_times = [results_by_size[s]['solve_time'] for s in sizes]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Scalability Analysis of Combined IGP/MPLS-TE Routing', 
                     fontsize=16)
        
        # 1. Network size growth
        ax1 = axes[0, 0]
        ax1.plot(sizes, edges, 'b-o', markersize=8)
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Number of Edges')
        ax1.set_title('Network Complexity Growth')
        ax1.grid(True, alpha=0.3)
        
        # 2. Utilization comparison
        ax2 = axes[0, 1]
        ax2.plot(sizes, igp_utils, 'r-s', label='IGP Only', markersize=8)
        ax2.plot(sizes, combined_utils, 'g-^', label='Combined', markersize=8)
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Max Link Utilization (%)')
        ax2.set_title('Utilization Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Improvement percentage
        ax3 = axes[0, 2]
        ax3.plot(sizes, improvements, 'purple', marker='d', markersize=8)
        ax3.set_xlabel('Number of Nodes')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Utilization Improvement')
        ax3.grid(True, alpha=0.3)
        
        # 4. LSP scaling
        ax4 = axes[1, 0]
        ax4.plot(sizes, lsp_counts, 'orange', marker='o', markersize=8)
        ax4.set_xlabel('Number of Nodes')
        ax4.set_ylabel('Number of LSPs')
        ax4.set_title('LSP Complexity')
        ax4.grid(True, alpha=0.3)
        
        # 5. Computational time
        ax5 = axes[1, 1]
        ax5.semilogy(sizes, solve_times, 'brown', marker='s', markersize=8)
        ax5.set_xlabel('Number of Nodes')
        ax5.set_ylabel('Solve Time (seconds)')
        ax5.set_title('Computational Complexity')
        ax5.grid(True, alpha=0.3)
        
        # 6. Efficiency metric
        ax6 = axes[1, 2]
        efficiency = [imp/time for imp, time in zip(improvements, solve_times)]
        ax6.plot(sizes, efficiency, 'darkgreen', marker='*', markersize=10)
        ax6.set_xlabel('Number of Nodes')
        ax6.set_ylabel('Improvement per Second (%/s)')
        ax6.set_title('Optimization Efficiency')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scalability_analysis_improved.png', dpi=150)
        plt.show()
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Average IGP utilization: {np.mean(igp_utils):.1f}%")
        print(f"Average combined utilization: {np.mean(combined_utils):.1f}%")
        print(f"Average improvement: {np.mean(improvements):.1f}%")
        print(f"Max solve time: {max(solve_times):.3f}s")
        
        # Complexity analysis
        if len(sizes) >= 4:
            log_n = np.log(sizes)
            log_t = np.log(solve_times)
            complexity = np.polyfit(log_n, log_t, 1)[0]
            print(f"Time complexity: O(n^{complexity:.2f})")

def main():
    print("=== Improved Scalability Testing for Combined Routing ===\n")
    
    tester = ScalabilityTester()
    
    # Test 1: Size scaling
    results_by_size = tester.run_size_scaling_test(
        sizes=[8, 10, 12, 15, 18, 20, 25, 30],
        runs_per_size=3
    )
    
    # Test 2: Topology comparison
    tester.run_topology_comparison(n_nodes=20)
    
    # Test 3: Traffic pattern comparison  
    tester.run_traffic_pattern_comparison(n_nodes=20)
    
    # Plot results
    tester.plot_results(results_by_size)

if __name__ == "__main__":
    main()