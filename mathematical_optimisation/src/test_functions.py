import networkx as nx
from network_parsers import parse_atlanta_xml, parse_geant_xml
from igp_optimizer import IGPWeightOptimizer
from ecmp_handler import ECMPHandler
from iterative_optimizer import IterativeOptimizer
from combined_routing_optimizer import CombinedRoutingOptimizer
from visualization import (
    visualize_network_with_solution,
    visualize_failure_scenario,
    visualize_lsp_flows,
    visualize_igp_weights_comparison,
    visualize_simple_vs_iterative_comparison
)
def test_network_with_visualization(network_name, xml_path, parser_func):
    """Test a specific network with all visualizations"""
    print(f"\n{'='*70}")
    print(f"TESTING {network_name.upper()} NETWORK WITH VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Load network
    result = parser_func(xml_path)
    if result is None:
        print(f"Failed to load {network_name} network")
        return
    G, capacities, demands = result
    
    if G is None:
        print(f"Failed to load {network_name} network")
        return
    
    # Store position for consistent layouts
    pos = nx.kamada_kawai_layout(G, scale=2)
    
    # STEP 1: SIMPLE APPROACH (as in the paper)
    print("\n" + "="*60)
    print("STEP 1: SIMPLE APPROACH (Optimize metrics first, then LSPs)")
    print("As described in Cherubini et al. paper")
    print("="*60)
    
    # Default weights (all = 1)
    default_weights = {(u, v): 1 for u, v in G.edges()}
    default_weights.update({(v, u): 1 for u, v in G.edges()})
    
    # Optimize IGP weights
    print("\nOptimizing IGP weights with Tabu Search...")
    igp_optimizer = IGPWeightOptimizer(G, capacities, demands, max_iterations=50)
    optimized_weights = igp_optimizer.optimize()
    
    # Visualize the IGP weight comparison
    visualize_igp_weights_comparison(G, capacities, demands, default_weights, optimized_weights, pos)
    
    # Create routing matrix with optimized weights
    routing_matrix = ECMPHandler.compute_ecmp_routing(G, demands, optimized_weights)
    
    # Create optimizer
    optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix, use_ecmp=True)
    
    # Compute IGP-only utilization
    igp_util, bottleneck, arc_utils = optimizer.compute_igp_only_utilization()
    print(f"\nIGP-only Analysis:")
    print(f"  Maximum utilization: {igp_util:.1f}%")
    if bottleneck:
        print(f"  Bottleneck link: {bottleneck}")
    
    # Solve nominal case with simple approach
    print(f"\n{'='*60}")
    print("SOLVING NOMINAL CASE - Simple Approach")
    print("="*60)
    
    result_simple = optimizer.solve_nominal(delta=0, verbose=True)
    optimizer.analyze_solution(result_simple, igp_util)
    
    # Store simple approach results
    simple_util = result_simple['u_max_percent']
    simple_lsps = result_simple['lsp_count']
    
    # Visualize simple approach results
    visualize_network_with_solution(G, capacities, result_simple, optimizer, pos)
    visualize_lsp_flows(G, result_simple, optimizer, pos)
    
    # STEP 2: ITERATIVE APPROACH 
    print("\n" + "="*60)
    print("STEP 2: ITERATIVE APPROACH")
    print("="*60)
    
    iterative_opt = IterativeOptimizer(G, capacities, demands, max_iterations=3)
    iterative_result, iterative_weights = iterative_opt.optimize_iteratively(delta=1e-6, verbose=True)
    
    if iterative_result and iterative_result['status'] == 'optimal':
        print(f"\nITERATIVE OPTIMIZATION RESULTS:")
        print(f"   Max utilization: {iterative_result['u_max_percent']:.1f}%")
        print(f"   Number of LSPs: {iterative_result['lsp_count']}")
        print(f"   MPLS traffic: {iterative_result['mpls_percent']:.1f}%")
        
        # COMPARISON
        print(f"\nCOMPARISON: SIMPLE vs ITERATIVE")
        print(f"   Simple approach: {simple_util:.1f}% utilization, {simple_lsps} LSPs")
        print(f"   Iterative approach: {iterative_result['u_max_percent']:.1f}% utilization, {iterative_result['lsp_count']} LSPs")
        
        improvement = simple_util - iterative_result['u_max_percent']
        if improvement > 0:
            print(f"   → Iterative is {improvement:.1f}% better!")
        else:
            print(f"   → No improvement")
        
        # Visualize comparison
        visualize_simple_vs_iterative_comparison(network_name, result_simple, iterative_result, igp_util)
    
    # Solve with survivability
    print(f"\n{'='*60}")
    print("SOLVING WITH SURVIVABILITY (All single link failures)")
    print("="*60)
    
    result_survivable = optimizer.solve_with_survivability(delta=0, verbose=True)
    
    if result_survivable['status'] == 'optimal':
        optimizer.analyze_solution(result_survivable, igp_util)
        
        # Visualize survivability results
        visualize_network_with_solution(G, capacities, result_survivable, optimizer, pos)
        visualize_lsp_flows(G, result_survivable, optimizer, pos)
        
        # Test some critical failure scenarios
        critical_links = list(G.edges())[:3]
        print(f"\nTesting failure scenarios:")
        for failed_link in critical_links:
            print(f"  Visualizing failure of link {failed_link}...")
            visualize_failure_scenario(G, capacities, optimizer, failed_link, result_survivable, pos)
        
        # Show final comparison table
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS SUMMARY for {network_name}")
        print("="*60)
        print(f"\n{'Approach':25} | {'Working':>12} | {'Failure':>12} | {'LSPs':>8}")
        print(f"{'-'*25} | {'-'*12} | {'-'*12} | {'-'*8}")
        print(f"{'IGP-only':25} | {igp_util:11.1f}% | {'N/A':>11} | {'0':>8}")
        print(f"{'Simple (paper approach)':25} | {simple_util:11.1f}% | {result_survivable['u_max_percent']:11.1f}% | {simple_lsps:>8}")
        
        if iterative_result and iterative_result['status'] == 'optimal':
            print(f"{'Iterative (our enhance)':25} | {iterative_result['u_max_percent']:11.1f}% | {'N/A':>11} | {iterative_result['lsp_count']:>8}")
        
        print(f"\nSUCCESS: {network_name} network supports full survivability!")
       
    else:
       print(f"\nSurvivability failed for {network_name} network")
       print(f"Model status: {result_survivable.get('model_status', 'Unknown')}")

def test_complete_implementation():
   """Test the complete implementation with all components"""
   print("=== COMPLETE COMBINED IGP/MPLS-TE ROUTING OPTIMIZATION ===")
   print("Testing all components from Cherubini et al. (2011)")
   
   # Test on both Atlanta and Geant networks
   networks = [
       ("Atlanta", "data/atlanta.xml", parse_atlanta_xml),
       ("Geant", "data/geant.xml", parse_geant_xml)
   ]
   
   for net_name, xml_file, parser_func in networks:
       print(f"\n{'='*70}")
       print(f"TESTING {net_name.upper()} NETWORK")
       print(f"{'='*70}")
       
       # Load network
       result = parser_func(xml_file)
       if result is None:
           print(f"Failed to load {net_name} network")
           continue
       G, capacities, demands = result
       
       # 1. Test IGP weight optimization
       print(f"\n1. IGP Weight Optimization (Enhanced Tabu Search)")
       print("-" * 40)
       igp_optimizer = IGPWeightOptimizer(G, capacities, demands, max_iterations=50)
       optimized_weights = igp_optimizer.optimize()
       
       # 2. Test ECMP routing
       print(f"\n2. ECMP Routing Computation")
       print("-" * 40)
       ecmp_routing = ECMPHandler.compute_ecmp_routing(G, demands, optimized_weights)
       print(f"ECMP routing matrix computed with {len(ecmp_routing)} entries")
       
       # Count number of multi-path routes
       multi_path_count = 0
       for (s, t), demand in demands.items():
           arc_count = sum(1 for ((src, dst), arc) in ecmp_routing if src == s and dst == t)
           if arc_count > 1:
               multi_path_count += 1
       print(f"Multi-path routes: {multi_path_count} out of {len(demands)} commodities")
       
       # 3. Test iterative optimization
       print(f"\n3. Iterative Optimization Process")
       print("-" * 40)
       iterative_opt = IterativeOptimizer(G, capacities, demands, max_iterations=3)
       best_result, final_weights = iterative_opt.optimize_iteratively(
           initial_weights=optimized_weights, delta=1e-6, verbose=True
       )
       
       # 4. Test path decomposition
       if best_result and best_result['status'] == 'optimal':
           print(f"\n4. Path Decomposition")
           print("-" * 40)
           print(f"Decomposed {best_result['lsp_count']} LSP paths")
           if 'lsp_paths' in best_result:
               for i, (src, dst, path, flow) in enumerate(best_result['lsp_paths'][:3]):
                   print(f"  Path {i+1}: {' -> '.join(path)} ({flow:.1f} Mbps)")
       
       # 5. Test survivability with all components
       print(f"\n5. Full Survivability Test with ECMP")
       print("-" * 40)
       
       # Create optimizer with ECMP routing
       optimizer = CombinedRoutingOptimizer(G, capacities, demands, ecmp_routing, use_ecmp=True)
       
       # Solve with survivability
       survivable_result = optimizer.solve_with_survivability(delta=1e-6, verbose=True)
       
       if survivable_result['status'] == 'optimal':
           igp_util, _, _ = optimizer.compute_igp_only_utilization()
           optimizer.analyze_solution(survivable_result, igp_util)
       
       print(f"\n{'='*70}")
       print(f"COMPLETED TESTING FOR {net_name.upper()}")
       print(f"{'='*70}")