#!/usr/bin/env python3
"""
Scalability Testing for Combined IGP/MPLS-TE Routing
"""

import sys
import os

src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from scalability_tester import ScalabilityTester

def main():
    print("=== Scalability and Delta Parameter Analysis ===\n")
    
    tester = ScalabilityTester()

    print("\nPhase 0: Testing Real Networks from the Paper")
    real_network_results = tester.test_real_networks(delta=1e-6)

    print("\nPhase 0.5: Delta Analysis on Real Networks")
    real_delta_results = tester.run_delta_analysis_on_real_networks()

    # Test 1: Delta analysis for different network sizes
    print("\nPhase 1: Delta Parameter Analysis")
    delta_results_15 = tester.run_delta_analysis_for_size(n_nodes=15)
    delta_results_25 = tester.run_delta_analysis_for_size(n_nodes=25)
    
    # Test 2: Size scaling with optimal delta
    print("\nPhase 2: Network Size Scaling")
    results_by_size = tester.run_size_scaling_test(
        sizes=[5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
        runs_per_size=10
    )
    
    # Test 3: Topology comparison
    print("\nPhase 3: Topology Analysis")
    tester.run_topology_comparison(n_nodes=20)

    print("\nPhase 4: Direct Comparison Real vs Synthetic Networks")
    real_vs_syn = tester.compare_real_vs_synthetic(synthetic_size=15)
    
    # Plot scalability results
    tester.plot_results(results_by_size)

    print("\n" + "="*60)
    print("FINAL SUMMARY INCLUDING REAL NETWORKS")
    print("="*60)
    
    if real_network_results:
        print("\nReal Networks Performance:")
        for network, results in real_network_results.items():
            print(f"\n{network.upper()}:")
            print(f"  Improvement: {results['improvement']:.1f}%")
            print(f"  LSPs needed: {results['lsp_count']}")
            print(f"  Solve time: {results['solve_time']:.3f}s")

if __name__ == "__main__":
    main()
