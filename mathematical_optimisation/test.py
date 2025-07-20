#!/usr/bin/env python3
"""
Complete Implementation of Combined IGP/MPLS-TE Routing Optimization
from Cherubini et al. (2011)
"""

import sys
import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict


src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

print(f"Python path: {sys.path}")
print(f"Files in src: {os.listdir(src_path)}")



import network_parsers as np_module
import igp_optimizer as igp_opt
import path_decomposer as pd
import ecmp_handler as ecmp
import iterative_optimizer as it_opt
import combined_routing_optimizer as cro
import visualization as vis
import test_functions as tf

def main():
    """Main function to run complete implementation"""
    # Test all components
    tf.test_complete_implementation()
    
    # Test with visualizations for both networks
    print("\n\n" + "="*70)
    print("RUNNING TESTS WITH VISUALIZATIONS")
    print("="*70)
    
    # Test Atlanta network with visualizations
    tf.test_network_with_visualization("Atlanta", "data/atlanta.xml", np_module.parse_atlanta_xml)
    
    # Test Geant network with visualizations
    tf.test_network_with_visualization("Geant", "data/geant.xml", np_module.parse_geant_xml)
    
    # Also run the original test with the enhanced implementation
    print("\n\n" + "="*70)
    print("RUNNING ENHANCED ITERATIVE OPTIMIZATION")
    print("="*70)
    
    # Load Atlanta network
    result = np_module.parse_atlanta_xml("data/atlanta.xml")
    
    if result is not None:
        G, capacities, demands = result
        # Use the iterative optimizer for best results
        print("\nTesting iterative optimization with ECMP...")
        iterative_opt = it_opt.IterativeOptimizer(G, capacities, demands)
        best_result, final_weights = iterative_opt.optimize_iteratively(delta=1e-6, verbose=True)
        
        if best_result and best_result['status'] == 'optimal':
            print(f"\nFinal Results:")
            print(f"  Max utilization: {best_result['u_max_percent']:.1f}%")
            print(f"  Number of LSPs: {best_result['lsp_count']}")
            print(f"  IGP traffic: {best_result['igp_percent']:.1f}%")
            print(f"  MPLS traffic: {best_result['mpls_percent']:.1f}%")

if __name__ == "__main__":
    main()