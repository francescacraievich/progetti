#!/usr/bin/env python3
"""
Improved Combined IGP/MPLS-TE Routing Optimization
Better implementation following Cherubini et al. (2011) paper
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class CombinedRoutingOptimizer:
    def __init__(self, graph, capacities, demands, routing_matrix):
        """
        Initialize the optimizer
        
        Parameters:
        - graph: NetworkX graph representing the network topology
        - capacities: Dictionary {(i,j): capacity} for each link
        - demands: Dictionary {(s,t): demand} for each commodity
        - routing_matrix: IGP routing matrix X where X[f,(i,j)] is fraction of flow f on arc (i,j)
        """
        self.G = graph
        self.capacities = capacities
        self.demands = demands
        self.routing_matrix = routing_matrix
        
        # Extract network components
        self.nodes = list(graph.nodes())
        self.arcs = [(i,j) for i,j in graph.edges()] + [(j,i) for i,j in graph.edges()]
        self.commodities = list(demands.keys())
        
        # Aggregate commodities by origin node
        self.origin_nodes = list(set([s for s,t in self.commodities]))
        self.commodities_by_origin = {v: [(s,t) for s,t in self.commodities if s == v] 
                                     for v in self.origin_nodes}
    
    def compute_igp_only_utilization(self):
        """Calculate max utilization with pure IGP routing"""
        arc_loads = {}
        for arc in self.arcs:
            load = 0
            for f in self.commodities:
                if (f, arc) in self.routing_matrix:
                    load += self.demands[f] * self.routing_matrix[f, arc]
            arc_loads[arc] = load
        
        max_util = 0
        bottleneck_arc = None
        arc_utils = {}
        
        for arc in self.arcs:
            if arc in self.capacities and self.capacities[arc] > 0:
                util = arc_loads[arc] / self.capacities[arc]
                arc_utils[arc] = util * 100
                if util > max_util:
                    max_util = util
                    bottleneck_arc = arc
        
        # Debug: print all arc utilizations if needed
        if False:  # Set to True for debugging
            print("\nDebug - All arc utilizations:")
            for arc, util in sorted(arc_utils.items(), key=lambda x: x[1], reverse=True):
                if util > 0:
                    print(f"  {arc}: {util:.1f}% (load={arc_loads[arc]:.0f})")
        
        return max_util * 100, bottleneck_arc, arc_utils
        
    def solve_nominal(self, delta=1e-6, verbose=False):
        """
        Solve the nominal (no failure) combined routing problem
        Using the exact formulation from the paper
        """
        model = gp.Model("Combined_Routing_Nominal")
        if not verbose:
            model.setParam('OutputFlag', 0)
        
        # Decision variables
        u_max = model.addVar(name="u_max", lb=0, ub=1)  # Max utilization as fraction
        
        # IS-IS/OSPF share variables (is^f in the paper)
        is_vars = {}
        for f in self.commodities:
            is_vars[f] = model.addVar(name=f"is_{f[0]}_{f[1]}", 
                                     lb=0, ub=self.demands[f])
        
        # MPLS-TE flow variables (w_ij^v in the paper)
        w_vars = {}
        for v in self.origin_nodes:
            for arc in self.arcs:
                w_vars[v, arc] = model.addVar(name=f"w_{v}_{arc[0]}_{arc[1]}", lb=0)
        
        # Objective function (Equation 6 in the paper with penalty for LSPs)
        obj = u_max
        if delta > 0:
            # Add penalty for each unit of flow leaving source nodes via MPLS
            for v in self.origin_nodes:
                for j in self.nodes:
                    if v != j and (v, j) in self.arcs:
                        obj += delta * w_vars[v, (v, j)]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        
        # Maximum utilization constraints (Equation 2)
        for arc in self.arcs:
            if arc in self.capacities and self.capacities[arc] > 0:
                expr = gp.LinExpr()
                
                # IS-IS/OSPF traffic
                for f in self.commodities:
                    if (f, arc) in self.routing_matrix:
                        expr += is_vars[f] * self.routing_matrix[f, arc]
                
                # MPLS-TE traffic
                for v in self.origin_nodes:
                    expr += w_vars[v, arc]
                
                model.addConstr(expr <= u_max * self.capacities[arc], 
                               name=f"util_{arc[0]}_{arc[1]}")
        
        # Flow conservation constraints for MPLS-TE (Equation 3)
        for v in self.origin_nodes:
            for i in self.nodes:
                expr = gp.LinExpr()
                
                # Incoming flow
                for j in self.nodes:
                    if (j, i) in self.arcs:
                        expr += w_vars[v, (j, i)]
                
                # Outgoing flow
                for j in self.nodes:
                    if (i, j) in self.arcs:
                        expr -= w_vars[v, (i, j)]
                
                # Right-hand side
                if i == v:  # Source node
                    # Sum of (is^f - d^f) for all commodities f originating at v
                    rhs = gp.LinExpr()
                    for f in self.commodities_by_origin[v]:
                        rhs += is_vars[f] - self.demands[f]
                    model.addConstr(expr == rhs, name=f"flow_cons_src_{v}_{i}")
                else:
                    # Check if i is destination for any commodity from v
                    rhs = gp.LinExpr()
                    for f in self.commodities_by_origin[v]:
                        if i == f[1]:  # i is destination of commodity f
                            rhs += self.demands[f] - is_vars[f]
                    model.addConstr(expr == rhs, name=f"flow_cons_{v}_{i}")
        
        # Optimize
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Extract results
            result = {
                'status': 'optimal',
                'u_max': u_max.X,
                'u_max_percent': u_max.X * 100,
                'model': model
            }
            
            # Get variable values
            result['is_values'] = {f: is_vars[f].X for f in self.commodities}
            result['w_values'] = {(v, arc): w_vars[v, arc].X 
                                 for v in self.origin_nodes 
                                 for arc in self.arcs 
                                 if w_vars[v, arc].X > 0.01}
            
            # Calculate traffic split
            total_demand = sum(self.demands.values())
            total_igp = sum(result['is_values'].values())
            result['total_igp'] = total_igp
            result['total_mpls'] = total_demand - total_igp
            result['igp_percent'] = (total_igp / total_demand * 100) if total_demand > 0 else 0
            result['mpls_percent'] = 100 - result['igp_percent']
            
            # Count LSPs (one per source with positive outgoing flow)
            lsp_count = 0
            lsp_details = []
            for v in self.origin_nodes:
                for j in self.nodes:
                    if v != j and (v, j) in self.arcs and w_vars[v, (v, j)].X > 0.01:
                        lsp_count += 1
                        lsp_details.append((v, j, w_vars[v, (v, j)].X))
            
            result['lsp_count'] = lsp_count
            result['lsp_details'] = sorted(lsp_details, key=lambda x: x[2], reverse=True)
            
            return result
        else:
            return {'status': 'infeasible'}
    
    def solve_with_survivability(self, failure_scenarios=None, delta=1e-6):
        """
        Solve with survivability constraints (simplified version)
        In full implementation, would need failure-specific routing matrices
        """
        # For now, just return nominal solution
        # Full implementation would add constraints from Equation 7 in the paper
        return self.solve_nominal(delta)
    
    def analyze_solution(self, result, igp_util):
        """Analyze and print detailed solution information"""
        if result['status'] != 'optimal':
            print("No optimal solution found!")
            return
        
        print(f"\n{'='*60}")
        print("SOLUTION ANALYSIS")
        print(f"{'='*60}")
        
        # Utilization improvement
        improvement = ((igp_util - result['u_max_percent']) / igp_util * 100) if igp_util > 0 else 0
        print(f"\nNetwork Utilization:")
        print(f"  IGP-only max utilization: {igp_util:.1f}%")
        print(f"  Combined max utilization: {result['u_max_percent']:.1f}%")
        print(f"  Improvement: {improvement:.1f}%")
        
        # Traffic split
        print(f"\nTraffic Distribution:")
        print(f"  Total demand: {sum(self.demands.values()):.0f} Mbps")
        print(f"  IGP traffic: {result['total_igp']:.0f} Mbps ({result['igp_percent']:.1f}%)")
        print(f"  MPLS traffic: {result['total_mpls']:.0f} Mbps ({result['mpls_percent']:.1f}%)")
        
        # LSP information
        print(f"\nLSP Information:")
        print(f"  Number of LSPs: {result['lsp_count']}")
        if result['lsp_details']:
            print(f"  Top LSPs by traffic volume:")
            for src, next_hop, flow in result['lsp_details'][:5]:
                print(f"    LSP from {src} to {next_hop}: {flow:.1f} Mbps")

def create_test_instance():
    """Create the test instance from the paper (Figure 2)"""
    # Create graph - ring topology with cross-links
    G = nx.Graph()
    nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    G.add_nodes_from(nodes)
    
    # Edges exactly as shown in Figure 2 of the paper
    edges = [('a', 'b'), ('b', 'd'), ('d', 'g'), ('g', 'e'), 
             ('e', 'h'), ('h', 'f'), ('f', 'c'), ('c', 'a'),  # ring
             ('c', 'd'), ('f', 'g')]  # cross-links
    
    capacities = {}
    for i, j in edges:
        G.add_edge(i, j)
        capacities[(i,j)] = 1000
        capacities[(j,i)] = 1000
    
    # Traffic matrix from Figure 2 (complete matrix)
    traffic_matrix = {
        ('a', 'b'): 40, ('a', 'c'): 30, ('a', 'd'): 60, ('a', 'e'): 20,
        ('a', 'f'): 10, ('a', 'g'): 70, ('a', 'h'): 70,
        ('b', 'a'): 40, ('b', 'c'): 60, ('b', 'd'): 20, ('b', 'e'): 70,
        ('b', 'f'): 40, ('b', 'g'): 50, ('b', 'h'): 50,
        ('c', 'a'): 80, ('c', 'b'): 70, ('c', 'd'): 60, ('c', 'e'): 40,
        ('c', 'f'): 80, ('c', 'g'): 10, ('c', 'h'): 60,
        ('d', 'a'): 30, ('d', 'b'): 80, ('d', 'c'): 100, ('d', 'e'): 40,
        ('d', 'f'): 60, ('d', 'g'): 70, ('d', 'h'): 120,
        ('e', 'a'): 20, ('e', 'b'): 60, ('e', 'c'): 10, ('e', 'd'): 50,
        ('e', 'f'): 20, ('e', 'g'): 60, ('e', 'h'): 30,
        ('f', 'a'): 30, ('f', 'b'): 50, ('f', 'c'): 70, ('f', 'd'): 80,
        ('f', 'e'): 30, ('f', 'g'): 100, ('f', 'h'): 40,
        ('g', 'a'): 20, ('g', 'b'): 100, ('g', 'c'): 150, ('g', 'd'): 50,
        ('g', 'e'): 60, ('g', 'f'): 80, ('g', 'h'): 60,
        ('h', 'a'): 100, ('h', 'b'): 100, ('h', 'c'): 150, ('h', 'd'): 200,
        ('h', 'e'): 50, ('h', 'f'): 100, ('h', 'g'): 100
    }
    
    # Compute shortest path routing
    routing_matrix = compute_ospf_routing(G, traffic_matrix)
    
    return G, capacities, traffic_matrix, routing_matrix

def compute_ospf_routing(G, demands):
    """Compute OSPF routing matrix using shortest paths"""
    routing_matrix = {}
    
    for (s, t), demand in demands.items():
        try:
            path = nx.shortest_path(G, s, t)
            for i in range(len(path) - 1):
                arc = (path[i], path[i+1])
                routing_matrix[(s,t), arc] = 1.0
        except nx.NetworkXNoPath:
            pass
    
    return routing_matrix

def visualize_network_with_solution(G, capacities, result, optimizer):
    """Enhanced visualization showing traffic flows and utilization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    pos = nx.circular_layout(G)
    
    # Calculate IGP-only utilizations for comparison
    igp_util, _, igp_arc_utils = optimizer.compute_igp_only_utilization()
    
    # Left plot: IGP-only utilization
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1500, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax1)
    
    # Color edges based on IGP utilization
    edge_colors_igp = []
    edge_widths_igp = []
    for u, v in G.edges():
        # Get max utilization for this edge (both directions)
        util1 = igp_arc_utils.get((u, v), 0)
        util2 = igp_arc_utils.get((v, u), 0)
        max_util = max(util1, util2)
        
        # Color coding: green < 50%, yellow 50-80%, orange 80-100%, red > 100%
        if max_util < 50:
            edge_colors_igp.append('green')
        elif max_util < 80:
            edge_colors_igp.append('yellow')
        elif max_util < 100:
            edge_colors_igp.append('orange')
        else:
            edge_colors_igp.append('red')
        
        # Width based on utilization
        edge_widths_igp.append(1 + max_util / 20)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors_igp, 
                          width=edge_widths_igp, ax=ax1)
    
    # Add utilization labels
    edge_labels_igp = {}
    for u, v in G.edges():
        util1 = igp_arc_utils.get((u, v), 0)
        util2 = igp_arc_utils.get((v, u), 0)
        max_util = max(util1, util2)
        if max_util > 0:
            edge_labels_igp[(u, v)] = f"{max_util:.0f}%"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels_igp, font_size=10, ax=ax1)
    
    ax1.set_title(f"IGP-Only Routing (Max Util: {igp_util:.1f}%)", fontsize=14)
    ax1.axis('off')
    
    # Add legend for IGP
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='< 50%'),
        Patch(facecolor='yellow', label='50-80%'),
        Patch(facecolor='orange', label='80-100%'),
        Patch(facecolor='red', label='> 100%')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Right plot: Combined IGP/MPLS solution
    if result['status'] == 'optimal':
        # Calculate combined utilizations
        combined_arc_utils = {}
        
        # First, add IGP traffic
        for arc in optimizer.arcs:
            load = 0
            for f in optimizer.commodities:
                if (f, arc) in optimizer.routing_matrix:
                    load += result['is_values'][f] * optimizer.routing_matrix[f, arc]
            combined_arc_utils[arc] = load
        
        # Then add MPLS traffic
        for (v, arc), flow in result['w_values'].items():
            combined_arc_utils[arc] = combined_arc_utils.get(arc, 0) + flow
        
        # Convert to utilization percentages
        for arc in combined_arc_utils:
            if arc in capacities and capacities[arc] > 0:
                combined_arc_utils[arc] = (combined_arc_utils[arc] / capacities[arc]) * 100
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                              node_size=1500, ax=ax2)
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax2)
        
        # Color edges based on combined utilization
        edge_colors_combined = []
        edge_widths_combined = []
        for u, v in G.edges():
            util1 = combined_arc_utils.get((u, v), 0)
            util2 = combined_arc_utils.get((v, u), 0)
            max_util = max(util1, util2)
            
            if max_util < 50:
                edge_colors_combined.append('green')
            elif max_util < 80:
                edge_colors_combined.append('yellow')
            elif max_util < 100:
                edge_colors_combined.append('orange')
            else:
                edge_colors_combined.append('red')
            
            edge_widths_combined.append(1 + max_util / 20)
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors_combined,
                              width=edge_widths_combined, ax=ax2)
        
        # Add utilization labels
        edge_labels_combined = {}
        for u, v in G.edges():
            util1 = combined_arc_utils.get((u, v), 0)
            util2 = combined_arc_utils.get((v, u), 0)
            max_util = max(util1, util2)
            if max_util > 0:
                edge_labels_combined[(u, v)] = f"{max_util:.0f}%"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels_combined, font_size=10, ax=ax2)
        
        ax2.set_title(f"Combined Solution (Max Util: {result['u_max_percent']:.1f}%)", 
                     fontsize=14)
        ax2.legend(handles=legend_elements, loc='upper right')
    else:
        ax2.text(0.5, 0.5, "No Solution Found", ha='center', va='center', 
                fontsize=20, transform=ax2.transAxes)
    
    ax2.axis('off')
    
    # Add overall statistics
    fig.text(0.5, 0.02, f"Total Demand: {sum(optimizer.demands.values()):.0f} Mbps | "
                       f"IGP Traffic: {result['total_igp']:.0f} Mbps ({result['igp_percent']:.1f}%) | "
                       f"MPLS Traffic: {result['total_mpls']:.0f} Mbps ({result['mpls_percent']:.1f}%) | "
                       f"LSPs: {result['lsp_count']}",
            ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()

def test_different_delta_values():
    """Test the effect of different delta values on the solution"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT DELTA VALUES")
    print("="*60)
    
    G, capacities, demands, routing_matrix = create_test_instance()
    optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
    
    # Test different delta values
    delta_values = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    
    print(f"\n{'Delta':>10} | {'Max Util %':>10} | {'LSPs':>6} | {'IGP %':>6} | {'MPLS %':>7}")
    print("-" * 50)
    
    for delta in delta_values:
        result = optimizer.solve_nominal(delta=delta)
        if result['status'] == 'optimal':
            print(f"{delta:10.0e} | {result['u_max_percent']:10.1f} | "
                  f"{result['lsp_count']:6d} | {result['igp_percent']:6.1f} | "
                  f"{result['mpls_percent']:7.1f}")

def visualize_lsp_flows(G, result, optimizer):
    """Visualize the LSP flows in the network"""
    if result['status'] != 'optimal' or result['lsp_count'] == 0:
        print("No LSP flows to visualize")
        return
    
    plt.figure(figsize=(12, 10))
    pos = nx.circular_layout(G)
    
    # Draw base network
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5)
    
    # Highlight source nodes with MPLS traffic
    source_nodes = set()
    for v in optimizer.origin_nodes:
        for arc in optimizer.arcs:
            if arc[0] == v and (v, arc) in result['w_values'] and result['w_values'][(v, arc)] > 0.01:
                source_nodes.add(v)
    
    nx.draw_networkx_nodes(G, pos, nodelist=list(source_nodes), 
                          node_color='red', node_size=2000)
    
    # Draw LSP flows
    lsp_edges = []
    lsp_labels = {}
    
    for (v, arc), flow in result['w_values'].items():
        if arc[0] == v and flow > 0.01:  # Source edge of LSP
            lsp_edges.append(arc)
            lsp_labels[arc] = f"{flow:.0f}"
    
    # Draw LSP edges with different colors
    nx.draw_networkx_edges(G, pos, edgelist=lsp_edges, 
                          edge_color='red', width=3, 
                          arrows=True, arrowsize=20, arrowstyle='->')
    
    # Add flow labels
    nx.draw_networkx_edge_labels(G, pos, lsp_labels, font_size=10, 
                                label_pos=0.3, font_color='red')
    
    plt.title(f"LSP Flows (Total {result['lsp_count']} LSPs, {result['total_mpls']:.0f} Mbps)", 
             fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("=== Improved Combined IGP/MPLS-TE Routing Optimization ===")
    print("Based on Cherubini et al. (2011)")
    
    # Create test instance
    G, capacities, demands, routing_matrix = create_test_instance()
    
    print(f"\nNetwork Configuration:")
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Links: {len(G.edges())}")
    print(f"  Commodities: {len(demands)}")
    print(f"  Total demand: {sum(demands.values())} Mbps")
    print(f"  Total capacity: {sum(capacities.values())/2} Mbps")
    
    # Note about capacities
    print(f"\nNote: All link capacities are 1000 Mbps as specified in the paper")
    
    # Create optimizer
    optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
    
    # Compute IGP-only utilization
    igp_util, bottleneck, arc_utils = optimizer.compute_igp_only_utilization()
    print(f"\nIGP-only Analysis:")
    print(f"  Maximum utilization: {igp_util:.1f}%")
    if bottleneck:
        print(f"  Bottleneck link: {bottleneck}")
    
    # Show top congested links
    if igp_util > 100:
        print(f"  Congested links (>100% utilization):")
        sorted_utils = sorted(arc_utils.items(), key=lambda x: x[1], reverse=True)
        for arc, util in sorted_utils[:5]:
            if util > 100:
                print(f"    {arc}: {util:.1f}%")
    
    # Test with different delta values
    test_different_delta_values()
    
    # Solve with recommended delta
    print(f"\n{'='*60}")
    print("SOLVING WITH RECOMMENDED PARAMETERS")
    print("="*60)
    
    # Use delta=0 for maximum MPLS usage as suggested by the paper results
    result = optimizer.solve_nominal(delta=0, verbose=True)
    
    # Analyze solution
    optimizer.analyze_solution(result, igp_util)
    
    # Visualize utilization comparison
    visualize_network_with_solution(G, capacities, result, optimizer)
    
    # Visualize LSP flows
    visualize_lsp_flows(G, result, optimizer)

if __name__ == "__main__":
    main()