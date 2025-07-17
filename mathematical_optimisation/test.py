#!/usr/bin/env python3
"""
Complete Combined IGP/MPLS-TE Routing Optimization
Implementation of Cherubini et al. (2011) paper including survivability
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

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
        
        return max_util * 100, bottleneck_arc, arc_utils
    
    def compute_failure_routing_matrix(self, failed_link):
        """Compute routing matrix when a link fails"""
        # Create temporary graph without failed link
        G_temp = self.G.copy()
        
        # Remove both directions of the failed link
        if G_temp.has_edge(failed_link[0], failed_link[1]):
            G_temp.remove_edge(failed_link[0], failed_link[1])
        
        # Compute new shortest paths
        failure_routing_matrix = {}
        
        for (s, t), demand in self.demands.items():
            try:
                # Find shortest path in failed network
                path = nx.shortest_path(G_temp, s, t)
                # Assign flow to path edges
                for i in range(len(path) - 1):
                    arc = (path[i], path[i+1])
                    failure_routing_matrix[(s,t), arc] = 1.0
            except nx.NetworkXNoPath:
                # No path exists after failure - this commodity cannot be routed
                pass
        
        return failure_routing_matrix
        
    def solve_nominal(self, delta=0, verbose=False):
        """
        Solve the nominal (no failure) combined routing problem
        Using Equation 6 as objective function
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
        
        # Objective function (Equation 6 in the paper)
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
                'model': model,
                'objective_value': model.objVal
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
            
            # Count LSPs
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
    
    def solve_with_survivability(self, failure_scenarios=None, delta=1e-7, verbose=False):
        """
        Solve with survivability constraints (Equation 7) - COMPLETE implementation
        """
        model = gp.Model("Combined_Routing_Survivable")
        if not verbose:
            model.setParam('OutputFlag', 0)
        
        # If no failure scenarios specified, consider all single link failures
        if failure_scenarios is None:
            failure_scenarios = list(self.G.edges())
        
        # Precompute failure routing matrices
        print(f"Precomputing routing for {len(failure_scenarios)} failure scenarios...")
        failure_routing_matrices = {}
        for link in failure_scenarios:
            failure_routing_matrices[link] = self.compute_failure_routing_matrix(link)
        
        # Decision variables
        u_max = model.addVar(name="u_max", lb=0, ub=1)
        
        # IS-IS/OSPF share variables
        is_vars = {}
        for f in self.commodities:
            is_vars[f] = model.addVar(name=f"is_{f[0]}_{f[1]}", 
                                     lb=0, ub=self.demands[f])
        
        # MPLS-TE flow variables
        w_vars = {}
        for v in self.origin_nodes:
            for arc in self.arcs:
                w_vars[v, arc] = model.addVar(name=f"w_{v}_{arc[0]}_{arc[1]}", lb=0)
        
        # COMPLETE IMPLEMENTATION: Rerouting fractions for MPLS traffic
        # These represent how MPLS traffic is distributed when a link fails
        x_reroute_vars = {}
        for failed_link in failure_scenarios:
            # For flow on the failed link in both directions
            for direction in ['plus', 'minus']:
                # Create variables for how this flow is redistributed
                for arc in self.arcs:
                    # Skip the failed arc itself
                    if arc == failed_link or arc == (failed_link[1], failed_link[0]):
                        continue
                    
                    x_reroute_vars[(failed_link, direction, arc)] = model.addVar(
                        name=f"x_reroute_{failed_link[0]}_{failed_link[1]}_{direction}_{arc[0]}_{arc[1]}", 
                        lb=0, ub=1
                    )
        
        # Objective function (Equation 6)
        obj = u_max
        if delta > 0:
            for v in self.origin_nodes:
                for j in self.nodes:
                    if v != j and (v, j) in self.arcs:
                        obj += delta * w_vars[v, (v, j)]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # NOMINAL CONSTRAINTS
        
        # Maximum utilization constraints for nominal case
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
                               name=f"nominal_util_{arc[0]}_{arc[1]}")
        
        # Flow conservation constraints
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
                    rhs = gp.LinExpr()
                    for f in self.commodities_by_origin[v]:
                        rhs += is_vars[f] - self.demands[f]
                    model.addConstr(expr == rhs, name=f"flow_cons_src_{v}_{i}")
                else:
                    rhs = gp.LinExpr()
                    for f in self.commodities_by_origin[v]:
                        if i == f[1]:
                            rhs += self.demands[f] - is_vars[f]
                    model.addConstr(expr == rhs, name=f"flow_cons_{v}_{i}")
        
        # SURVIVABILITY CONSTRAINTS (Complete Equation 7)
        for failed_link in failure_scenarios:
            failure_routing = failure_routing_matrices[failed_link]
            
            # Flow conservation for rerouting: flow from failed link must be redistributed
            # For l+ direction (failed_link[0] -> failed_link[1])
            if failed_link in self.arcs:
                # Total MPLS flow on l+
                total_flow_plus = gp.LinExpr()
                for v in self.origin_nodes:
                    total_flow_plus += w_vars[v, failed_link]
                
                # This flow must be redistributed
                redistrib_plus = gp.LinExpr()
                for arc in self.arcs:
                    if arc != failed_link and arc != (failed_link[1], failed_link[0]):
                        if (failed_link, 'plus', arc) in x_reroute_vars:
                            redistrib_plus += x_reroute_vars[(failed_link, 'plus', arc)]
                
                # Conservation: all flow must be redistributed
                model.addConstr(redistrib_plus == 1.0, 
                               name=f"reroute_conservation_{failed_link[0]}_{failed_link[1]}_plus")
            
            # For l- direction (failed_link[1] -> failed_link[0])
            failed_link_reverse = (failed_link[1], failed_link[0])
            if failed_link_reverse in self.arcs:
                # Total MPLS flow on l-
                total_flow_minus = gp.LinExpr()
                for v in self.origin_nodes:
                    total_flow_minus += w_vars[v, failed_link_reverse]
                
                # This flow must be redistributed
                redistrib_minus = gp.LinExpr()
                for arc in self.arcs:
                    if arc != failed_link and arc != failed_link_reverse:
                        if (failed_link, 'minus', arc) in x_reroute_vars:
                            redistrib_minus += x_reroute_vars[(failed_link, 'minus', arc)]
                
                # Conservation: all flow must be redistributed
                model.addConstr(redistrib_minus == 1.0, 
                               name=f"reroute_conservation_{failed_link[0]}_{failed_link[1]}_minus")
            
            # Capacity constraints under failure
            for arc in self.arcs:
                # Skip if arc is the failed link
                if arc == failed_link or arc == failed_link_reverse:
                    continue
                
                if arc in self.capacities and self.capacities[arc] > 0:
                    expr = gp.LinExpr()
                    
                    # IS-IS/OSPF traffic with failed routing
                    for f in self.commodities:
                        if (f, arc) in failure_routing:
                            expr += is_vars[f] * failure_routing[f, arc]
                    
                    # MPLS-TE traffic (non-failed)
                    for v in self.origin_nodes:
                        expr += w_vars[v, arc]
                    
                    # Rerouted MPLS traffic from l+
                    if (failed_link, 'plus', arc) in x_reroute_vars and failed_link in self.arcs:
                        rerouted_plus = gp.LinExpr()
                        for v in self.origin_nodes:
                            rerouted_plus += w_vars[v, failed_link]
                        expr += x_reroute_vars[(failed_link, 'plus', arc)] * rerouted_plus
                    
                    # Rerouted MPLS traffic from l-
                    if (failed_link, 'minus', arc) in x_reroute_vars and failed_link_reverse in self.arcs:
                        rerouted_minus = gp.LinExpr()
                        for v in self.origin_nodes:
                            rerouted_minus += w_vars[v, failed_link_reverse]
                        expr += x_reroute_vars[(failed_link, 'minus', arc)] * rerouted_minus
                    
                    # Must remain under capacity even with failure
                    model.addConstr(expr <= u_max * self.capacities[arc], 
                                   name=f"surv_{failed_link[0]}_{failed_link[1]}_{arc[0]}_{arc[1]}")
        
        # Optimize
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Extract results
            result = {
                'status': 'optimal',
                'u_max': u_max.X,
                'u_max_percent': u_max.X * 100,
                'model': model,
                'objective_value': model.objVal,
                'survivability': True,
                'num_failure_scenarios': len(failure_scenarios)
            }
            
            # Get variable values
            result['is_values'] = {f: is_vars[f].X for f in self.commodities}
            result['w_values'] = {(v, arc): w_vars[v, arc].X 
                                 for v in self.origin_nodes 
                                 for arc in self.arcs 
                                 if w_vars[v, arc].X > 0.01}
            
            # Store rerouting fractions for analysis
            result['x_reroute_values'] = {}
            for key, var in x_reroute_vars.items():
                if var.X > 0.01:
                    result['x_reroute_values'][key] = var.X
            
            # Calculate traffic split
            total_demand = sum(self.demands.values())
            total_igp = sum(result['is_values'].values())
            result['total_igp'] = total_igp
            result['total_mpls'] = total_demand - total_igp
            result['igp_percent'] = (total_igp / total_demand * 100) if total_demand > 0 else 0
            result['mpls_percent'] = 100 - result['igp_percent']

            # Count LSPs
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
    
    def analyze_solution(self, result, igp_util):
        """Analyze and print detailed solution information"""
        if result['status'] != 'optimal':
            print("No optimal solution found!")
            return
        
        print(f"\n{'='*60}")
        print("SOLUTION ANALYSIS")
        print(f"{'='*60}")
        
        # Check if this is a survivability solution
        if 'survivability' in result and result['survivability']:
            print(f"\nSurvivability: YES ({result['num_failure_scenarios']} failure scenarios)")
        else:
            print(f"\nSurvivability: NO (nominal case only)")
        
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

def visualize_network_with_solution(G, capacities, result, optimizer, pos=None):
    """Enhanced visualization showing traffic flows and utilization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    if pos is None:
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
        
        # Add title based on survivability
        if 'survivability' in result and result['survivability']:
            title = f"Combined Solution with Survivability (Max Util: {result['u_max_percent']:.1f}%)"
        else:
            title = f"Combined Solution (Max Util: {result['u_max_percent']:.1f}%)"
        ax2.set_title(title, fontsize=14)
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

def visualize_lsp_flows(G, result, optimizer, pos=None):
    """Visualize the LSP flows in the network"""
    if result['status'] != 'optimal' or result['lsp_count'] == 0:
        print("No LSP flows to visualize")
        return
    
    if pos is None:
        pos = nx.circular_layout(G)
    
    plt.figure(figsize=(14, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', font_color='white')
    
    # Draw base edges (thin, background)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5)
    
    # Extract LSP information directly from lsp_details
    edge_flows = {}
    lsp_info = []
    
    # Use the lsp_details from result which contains (src, next_hop, flow)
    for src, next_hop, flow in result['lsp_details']:
        lsp_info.append({
            'origin': src,
            'first_hop': next_hop,
            'flow': flow
        })
        
        # Add to edge flows (as undirected)
        edge = tuple(sorted([src, next_hop]))
        edge_flows[edge] = edge_flows.get(edge, 0) + flow
    
    # Draw edges with flows
    for edge, total_flow in edge_flows.items():
        if total_flow > 0:
            # Width proportional to flow
            width = min(1 + total_flow / 50, 8)
            
            # Draw edge
            nx.draw_networkx_edges(G, pos, [edge], 
                                 edge_color='red', 
                                 width=width, 
                                 alpha=0.7)
    
    # Add edge labels
    for edge, total_flow in edge_flows.items():
        if total_flow > 0:
            # Get positions
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            
            # Calculate midpoint with offset
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            
            # Add perpendicular offset
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                offset_x = -dy / length * 0.08
                offset_y = dx / length * 0.08
            else:
                offset_x = offset_y = 0
            
            plt.text(mx + offset_x, my + offset_y, f"{int(total_flow)}", 
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor='yellow', 
                             alpha=0.8, 
                             edgecolor='black'),
                    ha='center', va='center')
    
    # Create summary
    total_lsps = result['lsp_count']
    total_bandwidth = result['total_mpls']
    
    # Create legend text
    legend_text = [
        f"Total LSPs: {total_lsps}",
        f"Total MPLS Traffic: {total_bandwidth:.0f} Mbps",
        "─" * 25,
        "LSP Details:"
    ]
    
    # Show LSP details from result
    for i, (src, next_hop, flow) in enumerate(result['lsp_details'][:10], 1):
        legend_text.append(f"{i}. {src} → {next_hop}: {flow:.0f} Mbps")
    
    if total_lsps > 10:
        legend_text.append(f"... and {total_lsps - 10} more LSPs")
    
    # Add text box
    textstr = '\n'.join(legend_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top', bbox=props)
    
    # Add title based on survivability
    if 'survivability' in result and result['survivability']:
        title = f'LSP Flows with Survivability (Total {total_lsps} LSPs, {total_bandwidth:.0f} Mbps)'
    else:
        title = f'LSP Flows (Total {total_lsps} LSPs, {total_bandwidth:.0f} Mbps)'
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_failure_scenario(G, capacities, optimizer, failed_link, result_with_survivability, pos=None):
    """Visualize network utilization under a specific link failure with survivability solution"""
    if pos is None:
        pos = nx.circular_layout(G)
    
    plt.figure(figsize=(10, 8))
    
    # Compute failure routing matrix
    failure_routing = optimizer.compute_failure_routing_matrix(failed_link)
    
    # Calculate utilizations with failed link INCLUDING MPLS from survivability solution
    arc_utils_failure = {}
    
    for arc in optimizer.arcs:
        # Skip failed link
        if (arc == failed_link or arc == (failed_link[1], failed_link[0])):
            continue
        
        # IGP traffic with failure routing
        load = 0
        for f in optimizer.commodities:
            if (f, arc) in failure_routing:
                # Use the is_values from survivability solution
                load += result_with_survivability['is_values'][f] * failure_routing[f, arc]
        
        # Add MPLS traffic from survivability solution
        for (v, flow_arc), flow in result_with_survivability['w_values'].items():
            if flow_arc == arc:
                load += flow
        
        # Add rerouted MPLS traffic if we have rerouting information
        if 'x_reroute_values' in result_with_survivability:
            # Check for rerouted traffic from the failed link
            for (link, direction, dest_arc), fraction in result_with_survivability['x_reroute_values'].items():
                if link == failed_link and dest_arc == arc:
                    # Add the rerouted portion
                    if direction == 'plus':
                        # Traffic that was on failed_link
                        for (v, orig_arc), orig_flow in result_with_survivability['w_values'].items():
                            if orig_arc == failed_link:
                                load += orig_flow * fraction
                    elif direction == 'minus':
                        # Traffic that was on reverse of failed_link
                        for (v, orig_arc), orig_flow in result_with_survivability['w_values'].items():
                            if orig_arc == (failed_link[1], failed_link[0]):
                                load += orig_flow * fraction
        
        if arc in capacities and capacities[arc] > 0:
            arc_utils_failure[arc] = (load / capacities[arc]) * 100
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
    
    # Draw edges with failure scenario coloring
    for u, v in G.edges():
        if (u, v) == failed_link or (v, u) == failed_link:
            # Failed link in black/dashed
            nx.draw_networkx_edges(G, pos, [(u, v)], edge_color='black',
                                  style='dashed', width=3)
        else:
            # Get utilization
            util1 = arc_utils_failure.get((u, v), 0)
            util2 = arc_utils_failure.get((v, u), 0)
            max_util = max(util1, util2)
            
            # Color based on utilization
            if max_util < 50:
                color = 'green'
            elif max_util < 80:
                color = 'yellow'
            elif max_util < 100:
                color = 'orange'
            else:
                color = 'red'
            
            width = 1 + max_util / 20
            nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=color, width=width)
    
    # Add utilization labels
    edge_labels = {}
    for u, v in G.edges():
        if (u, v) == failed_link or (v, u) == failed_link:
            edge_labels[(u, v)] = "FAILED"
        else:
            util1 = arc_utils_failure.get((u, v), 0)
            util2 = arc_utils_failure.get((v, u), 0)
            max_util = max(util1, util2)
            if max_util > 0:
                edge_labels[(u, v)] = f"{max_util:.0f}%"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    # Find max utilization
    max_util_failure = max(arc_utils_failure.values()) if arc_utils_failure else 0
    
    plt.title(f"Network Under Failure with Survivability: Link {failed_link} Failed\n"
              f"Max Utilization: {max_util_failure:.1f}%", fontsize=16)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='< 50%'),
        Patch(facecolor='yellow', label='50-80%'),
        Patch(facecolor='orange', label='80-100%'),
        Patch(facecolor='red', label='> 100%'),
        Patch(facecolor='black', label='Failed Link')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
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

def test_survivability():
    """Test survivability constraints"""
    print("\n" + "="*60)
    print("TESTING SURVIVABILITY")
    print("="*60)
    
    G, capacities, demands, routing_matrix = create_test_instance()
    optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
    
    # Test with different numbers of failure scenarios
    print("\nTesting with subset of critical links:")
    critical_links = [('g', 'e'), ('d', 'g'), ('f', 'g')]  # High traffic links
    
    print(f"\nConsidering failures of: {critical_links}")
    
    start_time = time.time()
    result = optimizer.solve_with_survivability(
        failure_scenarios=critical_links,
        delta=1e-7,
        verbose=True
    )
    solve_time = time.time() - start_time
    
    if result['status'] == 'optimal':
        print(f"\nSolution found in {solve_time:.2f} seconds")
        print(f"Max utilization with survivability: {result['u_max_percent']:.1f}%")
        print(f"Number of LSPs: {result['lsp_count']}")
    else:
        print("No feasible solution with survivability constraints!")

def main():
    print("=== Complete Combined IGP/MPLS-TE Routing Optimization ===")
    print("Implementation with Survivability (Equation 7)")
    
    # Create test instance
    G, capacities, demands, routing_matrix = create_test_instance()
    
    print(f"\nNetwork Configuration:")
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Links: {len(G.edges())}")
    print(f"  Commodities: {len(demands)}")
    print(f"  Total demand: {sum(demands.values())} Mbps")
    print(f"  Total capacity: {sum(capacities.values())/2} Mbps")
    
    # Create optimizer
    optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
    
    # Compute IGP-only utilization
    igp_util, bottleneck, arc_utils = optimizer.compute_igp_only_utilization()
    print(f"\nIGP-only Analysis:")
    print(f"  Maximum utilization: {igp_util:.1f}%")
    if bottleneck:
        print(f"  Bottleneck link: {bottleneck}")
    
    # Store position for consistent layouts
    pos = nx.circular_layout(G)
    
    # Test with different delta values
    test_different_delta_values()
    
    # Solve nominal case with delta=0 as recommended by paper
    print(f"\n{'='*60}")
    print("SOLVING NOMINAL CASE (No Failures)")
    print("="*60)
    
    result_nominal = optimizer.solve_nominal(delta=0, verbose=True)
    optimizer.analyze_solution(result_nominal, igp_util)
    
    # Visualize nominal solution
    visualize_network_with_solution(G, capacities, result_nominal, optimizer, pos)
    
    # Visualize LSP flows for nominal
    visualize_lsp_flows(G, result_nominal, optimizer, pos)
    
    # Test survivability
    test_survivability()
    
    # Compare nominal vs survivable solutions
    print(f"\n{'='*60}")
    print("COMPARISON: NOMINAL vs SURVIVABLE")
    print("="*60)
    
    # Solve with all link failures with delta=1e-7
    print("\nSolving with ALL link failures considered...")
    start_time = time.time()
    result_survivable = optimizer.solve_with_survivability(delta=1e-7, verbose=False)
    solve_time = time.time() - start_time
    
    if result_survivable['status'] == 'optimal':
        print(f"\nSolution found in {solve_time:.2f} seconds")
        print(f"\nComparison:")
        print(f"  {'':20} | {'Nominal':>10} | {'Survivable':>12}")
        print(f"  {'-'*20} | {'-'*10} | {'-'*12}")
        print(f"  {'Max Utilization %':20} | {result_nominal['u_max_percent']:10.1f} | {result_survivable['u_max_percent']:12.1f}")
        print(f"  {'Number of LSPs':20} | {result_nominal['lsp_count']:10d} | {result_survivable['lsp_count']:12d}")
        print(f"  {'MPLS Traffic %':20} | {result_nominal['mpls_percent']:10.1f} | {result_survivable['mpls_percent']:12.1f}")
        print(f"  {'Objective Value':20} | {result_nominal['objective_value']:10.4f} | {result_survivable['objective_value']:12.4f}")
        
        # Visualize LSP flows for survivable
        visualize_lsp_flows(G, result_survivable, optimizer, pos)
        
        # Visualize a specific failure scenario WITH survivability solution
        print("\nVisualizing failure scenario with survivability: link ('g', 'd') fails...")
        visualize_failure_scenario(G, capacities, optimizer, ('g', 'd'), result_survivable, pos)
        
    else:
        print("No feasible solution with full survivability!")

if __name__ == "__main__":
    main()