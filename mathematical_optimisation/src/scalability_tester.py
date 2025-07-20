import time
import numpy as np
import matplotlib.pyplot as plt
from network_generator import NetworkGenerator
from delta_analyzer import DeltaAnalyzer
from combined_routing_optimizer import CombinedRoutingOptimizer
from network_parsers import parse_atlanta_xml, parse_geant_xml, compute_ospf_routing

class ScalabilityTester:
    def __init__(self):
        self.results = []
        self.generator = NetworkGenerator()
        self.delta_analyzer = DeltaAnalyzer()

    def test_real_networks(self, delta=0, verbose=True):
        """Test sulle reti reali del paper: Atlanta e Geant"""
        print("\n" + "="*60)
        print("TESTING REAL NETWORKS FROM THE PAPER")
        print("="*60)
        
        results = {}
        
        # Test Atlanta Network
        print("\n--- Testing Atlanta Network ---")
        atlanta_result = parse_atlanta_xml("data/atlanta.xml")
        if atlanta_result is not None:
            G_atlanta, capacities_atlanta, demands_atlanta = atlanta_result
            
            # routing matrix
            routing_matrix = compute_ospf_routing(G_atlanta, demands_atlanta)
            
            # optimizer
            optimizer = CombinedRoutingOptimizer(
                G_atlanta, capacities_atlanta, demands_atlanta, routing_matrix
            )
            
            # Calculate IGP-only utilization
            igp_util, _, _ = optimizer.compute_igp_only_utilization()
            
            # Resolve with combined routing
            start_time = time.time()
            result = optimizer.solve_nominal(delta=delta, verbose=False)
            solve_time = time.time() - start_time
            
            if result['status'] == 'optimal':
                results['atlanta'] = {
                    'nodes': len(G_atlanta.nodes()),
                    'edges': G_atlanta.number_of_edges(),
                    'commodities': len(demands_atlanta),
                    'igp_util': igp_util,
                    'combined_util': result['u_max_percent'],
                    'improvement': (igp_util - result['u_max_percent']) / igp_util * 100,
                    'lsp_count': result['lsp_count'],
                    'igp_percent': result['igp_percent'],
                    'mpls_percent': result['mpls_percent'],
                    'solve_time': solve_time,
                    'optimizer': optimizer
                }
                
                print(f"  Nodes: {results['atlanta']['nodes']}")
                print(f"  Edges: {results['atlanta']['edges']}")
                print(f"  Commodities: {results['atlanta']['commodities']}")
                print(f"  IGP utilization: {igp_util:.1f}%")
                print(f"  Combined utilization: {result['u_max_percent']:.1f}%")
                print(f"  Improvement: {results['atlanta']['improvement']:.1f}%")
                print(f"  LSPs: {result['lsp_count']}")
        
        # Test Geant Network
        print("\n--- Testing Geant Network ---")
        geant_result = parse_geant_xml("data/geant.xml")
        if geant_result is not None:
            G_geant, capacities_geant, demands_geant = geant_result
            
            
            routing_matrix = compute_ospf_routing(G_geant, demands_geant)
            
            
            optimizer = CombinedRoutingOptimizer(
                G_geant, capacities_geant, demands_geant, routing_matrix
            )
            
          
            igp_util, _, _ = optimizer.compute_igp_only_utilization()
            
           
            start_time = time.time()
            result = optimizer.solve_nominal(delta=delta, verbose=False)
            solve_time = time.time() - start_time
            
            if result['status'] == 'optimal':
                results['geant'] = {
                    'nodes': len(G_geant.nodes()),
                    'edges': G_geant.number_of_edges(),
                    'commodities': len(demands_geant),
                    'igp_util': igp_util,
                    'combined_util': result['u_max_percent'],
                    'improvement': (igp_util - result['u_max_percent']) / igp_util * 100,
                    'lsp_count': result['lsp_count'],
                    'igp_percent': result['igp_percent'],
                    'mpls_percent': result['mpls_percent'],
                    'solve_time': solve_time,
                    'optimizer': optimizer
                }
                
                print(f"  Nodes: {results['geant']['nodes']}")
                print(f"  Edges: {results['geant']['edges']}")
                print(f"  Commodities: {results['geant']['commodities']}")
                print(f"  IGP utilization: {igp_util:.1f}%")
                print(f"  Combined utilization: {result['u_max_percent']:.1f}%")
                print(f"  Improvement: {results['geant']['improvement']:.1f}%")
                print(f"  LSPs: {result['lsp_count']}")
        
        return results
    
    def run_delta_analysis_on_real_networks(self):
        """Analisi delta sulle reti reali del paper"""
        print("\n" + "="*60)
        print("DELTA ANALYSIS ON REAL NETWORKS")
        print("="*60)
        
        all_delta_results = {}
        
        # Test Atlanta
        print("\n--- Delta Analysis for Atlanta Network ---")
        atlanta_result = parse_atlanta_xml("data/atlanta.xml")
        if atlanta_result is not None:
            G, capacities, demands = atlanta_result
            routing_matrix = compute_ospf_routing(G, demands)
            optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
            igp_util, _, _ = optimizer.compute_igp_only_utilization()
            
            delta_results = self.delta_analyzer.test_different_delta_values(
                optimizer, igp_util, verbose=False
            )
            all_delta_results['atlanta'] = delta_results
            
            # Plot for Atlanta
            self.delta_analyzer.plot_delta_analysis(
                delta_results, 
                "Delta Analysis: Atlanta Network (Real)"
            )
        
        # Test Geant
        print("\n--- Delta Analysis for Geant Network ---")
        geant_result = parse_geant_xml("data/geant.xml")
        if geant_result is not None:
            G, capacities, demands = geant_result
            routing_matrix = compute_ospf_routing(G, demands)
            optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
            igp_util, _, _ = optimizer.compute_igp_only_utilization()
            
            delta_results = self.delta_analyzer.test_different_delta_values(
                optimizer, igp_util, verbose=False
            )
            all_delta_results['geant'] = delta_results
            
            # Plot for Geant
            self.delta_analyzer.plot_delta_analysis(
                delta_results, 
                "Delta Analysis: Geant Network (Real)"
            )
        
        return all_delta_results
    
    def test_instance(self, n_nodes, topology='waxman', traffic_pattern='gravity', 
                     delta=0, verbose=False):
        """Test a single instance"""
        
        # Generate network
        G = self.generator.create_topology(n_nodes, topology)
        capacities = self.generator.assign_capacities(G, 'realistic')
        
        # Generate traffic with appropriate intensity
        # Increase intensity to create more congestion
        intensity = 0.3 / np.sqrt(n_nodes / 15)
        demands = self.generator.generate_traffic_matrix(G, traffic_pattern, intensity)
        
        # Compute routing
        routing_matrix = compute_ospf_routing(G, demands)
        
        # Create optimizer
        optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
        
        # Compute IGP-only utilization
        igp_util, _, _ = optimizer.compute_igp_only_utilization()
        
        # Skip if IGP routing is not congested enough
        if igp_util < 50:  
            if verbose:
                print(f"  Skipping - IGP util too low: {igp_util:.1f}%")
            return None
        
        # Solve combined routing with specified delta
        start_time = time.time()
        result = optimizer.solve_nominal(delta=delta, verbose=False)
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
                'total_capacity': sum(capacities.values()) / 2,
                'optimizer': optimizer,  # Store optimizer for delta analysis
                'topology_type': topology  # Store topology type
            }
        else:
            return None
    
    def run_delta_analysis_for_size(self, n_nodes=20, topology='waxman', max_attempts=10):
        """Run delta analysis for a specific network size"""
        print(f"\n=== Delta Analysis for {n_nodes}-node {topology} network ===")
        
        # Try to generate a congested instance
        for attempt in range(max_attempts):
            result = self.test_instance(n_nodes, topology, delta=0, verbose=False)
            
            if result and result['igp_util'] >= 50:
                optimizer = result['optimizer']
                igp_util = result['igp_util']
                
                # Run delta analysis
                delta_results = self.delta_analyzer.test_different_delta_values(
                    optimizer, igp_util, verbose=False
                )
                
                # Plot results
                self.delta_analyzer.plot_delta_analysis(
                    delta_results, 
                    f"Delta Analysis: {n_nodes}-node {topology} network"
                )
                
                return delta_results
        
        # If I couldn't generate a congested instance after max_attempts
        print(f"Note: Generated instance with lower congestion after {max_attempts} attempts")
        # Use the last result anyway
        if result:
            optimizer = result['optimizer']
            igp_util = result['igp_util']
            delta_results = self.delta_analyzer.test_different_delta_values(
                optimizer, igp_util, verbose=False
            )
            self.delta_analyzer.plot_delta_analysis(
                delta_results, 
                f"Delta Analysis: {n_nodes}-node {topology} network (IGP util: {igp_util:.1f}%)"
            )
            return delta_results
        
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
                    if key != 'optimizer' and isinstance(size_results[0][key], (int, float)):
                        avg_result[key] = np.mean([r[key] for r in size_results])
                    elif key != 'optimizer':
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
    
    def compare_real_vs_synthetic(self, synthetic_size=15, topology_type='waxman'):
        """Confronta reti reali vs sintetiche di dimensione simile"""
        print("\n" + "="*60)
        print("COMPARING REAL NETWORKS VS SYNTHETIC")
        print("="*60)
        
        # Test real networks
        real_results = self.test_real_networks(delta=1e-6)
        
        # synthetic networks
        synthetic_results = {}
        
        # for Atlanta (15 nodes)
        print(f"\nGenerating synthetic network similar to Atlanta ({synthetic_size} nodes, {topology_type})...")
        result_syn_atlanta = self.test_instance(
            synthetic_size, topology=topology_type, traffic_pattern='gravity', delta=1e-6
        )
        if result_syn_atlanta:
            result_syn_atlanta['topology_type'] = topology_type  # Aggiungi il tipo
            synthetic_results['synthetic_atlanta'] = result_syn_atlanta
        
        # Per Geant (23 nodi)
        print(f"\nGenerating synthetic network similar to Geant (23 nodes, {topology_type})...")
        result_syn_geant = self.test_instance(
            23, topology=topology_type, traffic_pattern='gravity', delta=1e-6
        )
        if result_syn_geant:
            result_syn_geant['topology_type'] = topology_type  # Aggiungi il tipo
            synthetic_results['synthetic_geant'] = result_syn_geant
        
        # Comparison graph
        self._plot_real_vs_synthetic_comparison(real_results, synthetic_results)
        
        return real_results, synthetic_results
    
    def _plot_real_vs_synthetic_comparison(self, real_results, synthetic_results):
        """Crea grafico di confronto tra reti reali e sintetiche"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Real Networks (Paper) vs Synthetic Networks Comparison', fontsize=18, y=1.02)
        
        #data
        networks = []
        network_types = []  
        igp_utils = []
        combined_utils = []
        improvements = []
        lsp_counts = []
        
        # Add real data
        if 'atlanta' in real_results:
            networks.append('Atlanta\n(Real)')
            network_types.append('real')
            igp_utils.append(real_results['atlanta']['igp_util'])
            combined_utils.append(real_results['atlanta']['combined_util'])
            improvements.append(real_results['atlanta']['improvement'])
            lsp_counts.append(real_results['atlanta']['lsp_count'])
        
        # synthetic
        if 'synthetic_atlanta' in synthetic_results:
            topo = synthetic_results['synthetic_atlanta'].get('topology_type', 'waxman')
            networks.append(f'Synthetic\n15 nodes\n({topo.upper()})')
            network_types.append('synthetic')
            igp_utils.append(synthetic_results['synthetic_atlanta']['igp_util'])
            combined_utils.append(synthetic_results['synthetic_atlanta']['combined_util'])
            improvements.append(synthetic_results['synthetic_atlanta']['improvement'])
            lsp_counts.append(synthetic_results['synthetic_atlanta']['lsp_count'])
        
        if 'geant' in real_results:
            networks.append('Geant\n(Real)')
            network_types.append('real')
            igp_utils.append(real_results['geant']['igp_util'])
            combined_utils.append(real_results['geant']['combined_util'])
            improvements.append(real_results['geant']['improvement'])
            lsp_counts.append(real_results['geant']['lsp_count'])
        
        # synthetic
        if 'synthetic_geant' in synthetic_results:
            topo = synthetic_results['synthetic_geant'].get('topology_type', 'waxman')
            networks.append(f'Synthetic\n23 nodes\n({topo.upper()})')
            network_types.append('synthetic')
            igp_utils.append(synthetic_results['synthetic_geant']['igp_util'])
            combined_utils.append(synthetic_results['synthetic_geant']['combined_util'])
            improvements.append(synthetic_results['synthetic_geant']['improvement'])
            lsp_counts.append(synthetic_results['synthetic_geant']['lsp_count'])
        
        x = np.arange(len(networks))
        
        # Plot 1: Utilization comparison
        ax1 = axes[0]
        width = 0.35
        bars1 = ax1.bar(x - width/2, igp_utils, width, label='IGP Only', color='#e74c3c', alpha=0.8)
        bars2 = ax1.bar(x + width/2, combined_utils, width, label='Combined', color='#27ae60', alpha=0.8)
        
        
        for bar, val in zip(bars1, igp_utils):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
        
        for bar, val in zip(bars2, combined_utils):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
        
        ax1.set_ylabel('Max Utilization (%)', fontsize=12)
        ax1.set_title('Network Utilization Comparison', fontsize=14, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(networks, fontsize=10)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        
        max_util = max(max(igp_utils), max(combined_utils))
        ax1.set_ylim(0, max_util * 1.2)  # 20% di spazio extra
        
        # Plot 2: Improvement con colori diversi per real vs synthetic
        ax2 = axes[1]
        colors = []
        for i, net_type in enumerate(network_types):
            if net_type == 'real':
                colors.append('#3498db')  # Blu for real network
            else:
                colors.append('#9b59b6')  # Violet for synthetic
        
        bars = ax2.bar(x, improvements, color=colors, alpha=0.8)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_title('Utilization Improvement', fontsize=14, pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(networks, fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        
        max_imp = max(improvements)
        ax2.set_ylim(0, max_imp * 1.25)  
        
       
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max_imp * 0.02,
                    f'{imp:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3498db', alpha=0.8, label='Real Networks'),
                          Patch(facecolor='#9b59b6', alpha=0.8, label='Synthetic Networks')]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Plot 3: LSP count con colori coordinati
        ax3 = axes[2]
        bars = ax3.bar(x, lsp_counts, color=colors, alpha=0.8)
        ax3.set_ylabel('Number of LSPs', fontsize=12)
        ax3.set_title('LSP Complexity', fontsize=14, pad=20)
        ax3.set_xticks(x)
        ax3.set_xticklabels(networks, fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
     
        max_lsp = max(lsp_counts)
        ax3.set_ylim(0, max_lsp * 1.15)  
        
        for bar, count in zip(bars, lsp_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max_lsp * 0.02,
                    f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.05)  
        plt.savefig('real_vs_synthetic_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
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
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Scalability Analysis of Combined IGP/MPLS-TE Routing', 
                     fontsize=18, y=0.98)
        
        # 1. Network size growth
        ax1 = axes[0, 0]
        ax1.plot(sizes, edges, 'b-o', markersize=8)
        ax1.set_xlabel('Number of Nodes', fontsize=12)
        ax1.set_ylabel('Number of Edges', fontsize=12)
        ax1.set_title('Network Complexity Growth', fontsize=14, pad=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=10)
        
        # 2. Utilization comparison
        ax2 = axes[0, 1]
        ax2.plot(sizes, igp_utils, 'r-s', label='IGP Only', markersize=8)
        ax2.plot(sizes, combined_utils, 'g-^', label='Combined', markersize=8)
        ax2.set_xlabel('Number of Nodes', fontsize=12)
        ax2.set_ylabel('Max Link Utilization (%)', fontsize=12)
        ax2.set_title('Utilization Comparison', fontsize=14, pad=12)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=10)
        
        # 3. Improvement percentage
        ax3 = axes[0, 2]
        ax3.plot(sizes, improvements, 'purple', marker='d', markersize=8)
        ax3.set_xlabel('Number of Nodes', fontsize=12)
        ax3.set_ylabel('Improvement (%)', fontsize=12)
        ax3.set_title('Utilization Improvement', fontsize=14, pad=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=10)
        
        # 4. LSP scaling
        ax4 = axes[1, 0]
        ax4.plot(sizes, lsp_counts, 'orange', marker='o', markersize=8)
        ax4.set_xlabel('Number of Nodes', fontsize=12)
        ax4.set_ylabel('Number of LSPs', fontsize=12)
        ax4.set_title('LSP Complexity', fontsize=14, pad=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', labelsize=10)
        
        # 5. Computational time
        ax5 = axes[1, 1]
        ax5.semilogy(sizes, solve_times, 'brown', marker='s', markersize=8)
        ax5.set_xlabel('Number of Nodes', fontsize=12)
        ax5.set_ylabel('Solve Time (seconds)', fontsize=12)
        ax5.set_title('Computational Complexity', fontsize=14, pad=10)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', labelsize=10)
        
        # 6. Efficiency metric
        ax6 = axes[1, 2]
        efficiency = [imp/time for imp, time in zip(improvements, solve_times)]
        ax6.plot(sizes, efficiency, 'darkgreen', marker='*', markersize=10)
        ax6.set_xlabel('Number of Nodes', fontsize=12)
        ax6.set_ylabel('Improvement per Second (%/s)', fontsize=12)
        ax6.set_title('Optimization Efficiency', fontsize=14, pad=10)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
        plt.savefig('scalability_analysis.png', dpi=150, bbox_inches='tight')
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