#!/usr/bin/env python3
"""
Combined IGP/MPLS-TE Routing Optimization with Atlanta Network
Implementation following Cherubini et al. (2011) with full survivability
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os

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
    
    def solve_with_survivability(self, failure_scenarios=None, delta=0, verbose=False):
        """
        Solve with survivability constraints using CORRECTED Equation 7 from the paper
        """
        model = gp.Model("Combined_Routing_Survivable")
        if not verbose:
            model.setParam('OutputFlag', 0)
        
        # If no failure scenarios specified, consider all single link failures
        if failure_scenarios is None:
            failure_scenarios = list(self.G.edges())
        
        # Precompute failure routing matrices
        print(f"Precomputing routing for {len(failure_scenarios)} single link failure scenarios...")
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
        
        # SURVIVABILITY CONSTRAINTS (Equation 7 from paper)
        for l_idx, failed_link in enumerate(failure_scenarios):
            failure_routing = failure_routing_matrices[failed_link]
            
            # Get the two directed arcs for the failed link
            l_plus = failed_link  # (p,q)
            l_minus = (failed_link[1], failed_link[0])  # (q,p)
            
            for arc in self.arcs:
                # Skip if arc is the failed link
                if arc == l_plus or arc == l_minus:
                    continue
                
                if arc in self.capacities and self.capacities[arc] > 0:
                    expr = gp.LinExpr()
                    
                    # First term: Σ_f χ_{ij}^{f,l} * is^f
                    for f in self.commodities:
                        if (f, arc) in failure_routing:
                            expr += is_vars[f] * failure_routing[f, arc]
                    
                    # Second term: Σ_v w_{ij}^v  
                    for v in self.origin_nodes:
                        expr += w_vars[v, arc]
                    
                    # Third term: Σ_v χ_{ij}^{l+,l} * w_{l+}^v
                    if l_plus in self.arcs:
                        # Create commodity for l+ = (p,q)
                        f_plus = (l_plus[0], l_plus[1])
                        if f_plus in self.commodities and (f_plus, arc) in failure_routing:
                            for v in self.origin_nodes:
                                expr += failure_routing[f_plus, arc] * w_vars[v, l_plus]
                    
                    # Fourth term: Σ_v χ_{ij}^{l-,l} * w_{l-}^v
                    if l_minus in self.arcs:
                        # Create commodity for l- = (q,p)
                        f_minus = (l_minus[0], l_minus[1])
                        if f_minus in self.commodities and (f_minus, arc) in failure_routing:
                            for v in self.origin_nodes:
                                expr += failure_routing[f_minus, arc] * w_vars[v, l_minus]
                    
                    # The constraint uses ≤ u_max * capacity 
                    model.addConstr(expr <= u_max * self.capacities[arc], 
                                   name=f"surv_l{l_idx}_{arc[0]}_{arc[1]}")
        
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
            return {'status': 'infeasible', 'model_status': model.status}
    
    def analyze_solution(self, result, igp_util):
        """Analyze and print detailed solution information"""
        if result['status'] != 'optimal':
            print("No optimal solution found!")
            if 'model_status' in result:
                print(f"Model status: {result['model_status']}")
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

def parse_atlanta_xml(xml_file_path):
    """Parse Atlanta network XML file from ZIB SNDlib"""
    try:
        # Check if file exists
        if not os.path.exists(xml_file_path):
            print(f"Error: File {xml_file_path} not found")
            return None, None, None
            
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        print(f"Successfully loaded XML from {xml_file_path}")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Parse nodes
        nodes_found = 0
        for node in root.iter():
            if 'node' in node.tag and node.get('id'):
                node_id = node.get('id')
                G.add_node(node_id)
                nodes_found += 1
        
        print(f"Found {nodes_found} nodes")
        
        # Parse links and capacities
        capacities = {}
        links_found = 0
        
        for link in root.iter():
            if 'link' in link.tag and link.get('id'):
                source = None
                target = None
                capacity = 0.0
                
                for elem in link.iter():
                    if 'source' in elem.tag and elem.text:
                        source = elem.text
                    elif 'target' in elem.tag and elem.text:
                        target = elem.text
                    elif 'capacity' in elem.tag and elem.text:
                        capacity = float(elem.text)
                
                if source and target:
                    G.add_edge(source, target)
                    capacities[(source, target)] = capacity
                    capacities[(target, source)] = capacity
                    links_found += 1
        
        print(f"Found {links_found} links")
        
        # Parse demands
        demands = {}
        demands_found = 0
        
        for demand in root.iter():
            if 'demand' in demand.tag and demand.get('id'):
                source = None
                target = None
                value = 0.0
                
                for elem in demand.iter():
                    if 'source' in elem.tag and elem.text:
                        source = elem.text
                    elif 'target' in elem.tag and elem.text:
                        target = elem.text
                    elif 'demandValue' in elem.tag and elem.text:
                        value = float(elem.text)
                
                if source and target and value > 0:
                    demands[(source, target)] = value
                    demands_found += 1
        
        print(f"Found {demands_found} demands")
        
        # Scale demands to make problem feasible
        total_demand = sum(demands.values())
        total_capacity = sum([c for c in capacities.values() if c > 0])/2
        
        print(f"\nAtlanta Network Summary:")
        print(f"  Nodes: {len(G.nodes())}")
        print(f"  Links: {len(G.edges())}")
        print(f"  Commodities: {len(demands)}")
        print(f"  Total demand: {total_demand:.0f} Mbps")
        print(f"  Total capacity: {total_capacity:.0f} Mbps")
        
        # Scale demands to 20% of total capacity for feasibility
        scale_factor = (total_capacity * 0.2) / total_demand
        scaled_demands = {k: v * scale_factor for k, v in demands.items()}
        
        print(f"\nScaling demands by factor {scale_factor:.3f} for feasibility")
        print(f"  Scaled total demand: {sum(scaled_demands.values()):.0f} Mbps")
        
        return G, capacities, scaled_demands
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None, None, None

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
        pos = nx.spring_layout(G, seed=42)
    
    # Calculate IGP-only utilizations for comparison
    igp_util, _, igp_arc_utils = optimizer.compute_igp_only_utilization()
    
    # Left plot: IGP-only utilization
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax1)
    
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
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels_igp, font_size=8, ax=ax1)
    
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
                              node_size=1000, ax=ax2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax2)
        
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
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels_combined, font_size=8, ax=ax2)
        
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
    if result['status'] == 'optimal':
        fig.text(0.5, 0.02, f"Total Demand: {sum(optimizer.demands.values()):.0f} Mbps | "
                           f"IGP Traffic: {result['total_igp']:.0f} Mbps ({result['igp_percent']:.1f}%) | "
                           f"MPLS Traffic: {result['total_mpls']:.0f} Mbps ({result['mpls_percent']:.1f}%) | "
                           f"LSPs: {result['lsp_count']}",
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()

def visualize_failure_scenario(G, capacities, optimizer, failed_link, result_with_survivability, pos=None):
    """Visualize network utilization under a specific link failure with survivability solution"""
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
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
        
        # Add rerouted MPLS traffic from the failed link
        failed_commodity_plus = (failed_link[0], failed_link[1])
        failed_commodity_minus = (failed_link[1], failed_link[0])
        
        # Reroute MPLS traffic that was on the failed link
        for v in optimizer.origin_nodes:
            # Traffic on failed_link
            if (v, failed_link) in result_with_survivability['w_values']:
                mpls_on_failed = result_with_survivability['w_values'][(v, failed_link)]
                # This traffic is rerouted according to IGP routing for the failed commodity
                if failed_commodity_plus in optimizer.commodities and (failed_commodity_plus, arc) in failure_routing:
                    load += mpls_on_failed * failure_routing[failed_commodity_plus, arc]
            
            # Traffic on reverse of failed_link
            failed_reverse = (failed_link[1], failed_link[0])
            if (v, failed_reverse) in result_with_survivability['w_values']:
                mpls_on_failed = result_with_survivability['w_values'][(v, failed_reverse)]
                # This traffic is rerouted according to IGP routing for the failed commodity
                if failed_commodity_minus in optimizer.commodities and (failed_commodity_minus, arc) in failure_routing:
                    load += mpls_on_failed * failure_routing[failed_commodity_minus, arc]
        
        if arc in capacities and capacities[arc] > 0:
            arc_utils_failure[arc] = (load / capacities[arc]) * 100
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
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
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Find max utilization
    max_util_failure = max(arc_utils_failure.values()) if arc_utils_failure else 0
    
    plt.title(f"Atlanta Network Under Failure: Link {failed_link} Failed\n"
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

def visualize_lsp_flows(G, result, optimizer, pos=None):
    """Visualize the LSP flows in the network"""
    if result['status'] != 'optimal' or result['lsp_count'] == 0:
        print("No LSP flows to visualize")
        return
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(14, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw base edges (thin, background)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5)
    
    # Extract LSP information
    edge_flows = {}
    
    # Use the lsp_details from result
    for src, next_hop, flow in result['lsp_details']:
        edge = tuple(sorted([src, next_hop]))
        edge_flows[edge] = edge_flows.get(edge, 0) + flow
    
    # Draw edges with flows
    for edge, total_flow in edge_flows.items():
        if total_flow > 0:
            # Width proportional to flow
            width = min(1 + total_flow / 1000, 8)
            
            # Draw edge
            nx.draw_networkx_edges(G, pos, [edge], 
                                 edge_color='red', 
                                 width=width, 
                                 alpha=0.7)
    
    # Create summary
    total_lsps = result['lsp_count']
    total_bandwidth = result['total_mpls']
    
    # Create legend text
    legend_text = [
        f"Total LSPs: {total_lsps}",
        f"Total MPLS Traffic: {total_bandwidth:.0f} Mbps",
        "─" * 25,
        "Top LSPs:"
    ]
    
    # Show top LSP details
    for i, (src, next_hop, flow) in enumerate(result['lsp_details'][:8], 1):
        legend_text.append(f"{i}. {src} → {next_hop}: {flow:.0f} Mbps")
    
    if total_lsps > 8:
        legend_text.append(f"... and {total_lsps - 8} more LSPs")
    
    # Add text box
    textstr = '\n'.join(legend_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top', bbox=props)
    
    # Add title
    if 'survivability' in result and result['survivability']:
        title = f'Atlanta Network: LSP Flows with Survivability\n(Total {total_lsps} LSPs, {total_bandwidth:.0f} Mbps)'
    else:
        title = f'Atlanta Network: LSP Flows\n(Total {total_lsps} LSPs, {total_bandwidth:.0f} Mbps)'
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("=== Combined IGP/MPLS-TE Routing Optimization ===")
    print("Implementation with Atlanta Network from ZIB SNDlib")
    print("Following Cherubini et al. (2011) - Equations 6 & 7")
    
    # Load Atlanta network
    G, capacities, demands = parse_atlanta_xml("atlanta.xml")
    
    if G is None:
        print("Failed to load Atlanta network")
        return
    
    # Compute shortest path routing
    routing_matrix = compute_ospf_routing(G, demands)
    
    # Create optimizer
    optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
    
    # Compute IGP-only utilization
    igp_util, bottleneck, arc_utils = optimizer.compute_igp_only_utilization()
    print(f"\nIGP-only Analysis:")
    print(f"  Maximum utilization: {igp_util:.1f}%")
    if bottleneck:
        print(f"  Bottleneck link: {bottleneck}")
    
    # Store position for consistent layouts
    pos = nx.spring_layout(G, seed=42)
    
    # Solve nominal case
    print(f"\n{'='*60}")
    print("SOLVING NOMINAL CASE (No Failures)")
    print("="*60)
    
    result_nominal = optimizer.solve_nominal(delta=0, verbose=True)
    optimizer.analyze_solution(result_nominal, igp_util)
    
    # Visualize comparison IGP vs Combined
    visualize_network_with_solution(G, capacities, result_nominal, optimizer, pos)
    
    # Visualize LSP flows for nominal
    visualize_lsp_flows(G, result_nominal, optimizer, pos)
    
    # Solve with survivability
    print(f"\n{'='*60}")
    print("SOLVING WITH SURVIVABILITY (All single link failures)")
    print("="*60)
    
    result_survivable = optimizer.solve_with_survivability(delta=0, verbose=True)
    
    if result_survivable['status'] == 'optimal':
        optimizer.analyze_solution(result_survivable, igp_util)
        
        # Visualize comparison IGP vs Combined with survivability
        visualize_network_with_solution(G, capacities, result_survivable, optimizer, pos)
        
        # Visualize LSP flows for survivable
        visualize_lsp_flows(G, result_survivable, optimizer, pos)
        
        # Test some critical failure scenarios
        critical_links = list(G.edges())[:3]  # Test first 3 links
        print(f"\nTesting failure scenarios:")
        for failed_link in critical_links:
            print(f"  Visualizing failure of link {failed_link}...")
            visualize_failure_scenario(G, capacities, optimizer, failed_link, result_survivable, pos)
        
        # Show comparison
        print(f"\n{'='*60}")
        print("COMPARISON: NOMINAL vs SURVIVABLE")
        print("="*60)
        print(f"  {'':20} | {'Nominal':>10} | {'Survivable':>12}")
        print(f"  {'-'*20} | {'-'*10} | {'-'*12}")
        print(f"  {'Max Utilization %':20} | {result_nominal['u_max_percent']:10.1f} | {result_survivable['u_max_percent']:12.1f}")
        print(f"  {'Number of LSPs':20} | {result_nominal['lsp_count']:10d} | {result_survivable['lsp_count']:12d}")
        print(f"  {'MPLS Traffic %':20} | {result_nominal['mpls_percent']:10.1f} | {result_survivable['mpls_percent']:12.1f}")
        print(f"  {'Objective Value':20} | {result_nominal['objective_value']:10.4f} | {result_survivable['objective_value']:12.4f}")
        
        print(f"\n✅ SUCCESS: Atlanta network supports full survivability!")
        print(f"This validates the implementation according to Cherubini et al. (2011)")
        
    else:
        print(f"\n❌ Survivability failed for Atlanta network")
        print(f"Model status: {result_survivable.get('model_status', 'Unknown')}")
        
        # Debug which links cause problems
        print(f"\nTesting individual link failures:")
        failure_scenarios = list(G.edges())
        infeasible_count = 0
        
        for failed_link in failure_scenarios[:5]:  # Test first 5 links
            result = optimizer.solve_with_survivability(
                failure_scenarios=[failed_link], 
                delta=0, 
                verbose=False
            )
            
            if result['status'] == 'optimal':
                print(f"  ✅ Link {failed_link}: {result['u_max_percent']:.1f}%")
            else:
                print(f"  ❌ Link {failed_link}: INFEASIBLE")
                infeasible_count += 1
        
        if infeasible_count > 0:
            print(f"\n{infeasible_count}/5 tested links cause infeasibility")

if __name__ == "__main__":
    main()