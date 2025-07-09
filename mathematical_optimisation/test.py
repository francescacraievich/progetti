#!/usr/bin/env python3
"""
Combined IGP/MPLS-TE Routing Optimization
Based on Cherubini et al. (2011) paper
Small test instance implementation
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
        
    def solve_nominal(self, delta=0.0001):
        """
        Solve the nominal (no failure) combined routing problem
        
        Returns:
        - model: Gurobi model
        - u_max: Maximum link utilization
        - w_values: MPLS-TE flow values
        - is_values: IS-IS/OSPF flow shares
        """
        model = gp.Model("Combined_Routing_Nominal")
        
        # Variables
        u_max = model.addVar(name="u_max", lb=0)
        
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
        
        # Objective: minimize maximum utilization + delta * total MPLS flow
        obj = u_max
        if delta > 0:
            for v in self.origin_nodes:
                for neighbor in self.G.neighbors(v):
                    if (v, neighbor) in self.arcs:
                        obj += delta * w_vars[v, (v, neighbor)]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        
        # Maximum utilization constraints
        for arc in self.arcs:
            if arc in self.capacities:
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
        
        # Flow conservation constraints for MPLS-TE
        for v in self.origin_nodes:
            for i in self.nodes:
                expr = gp.LinExpr()
                
                # Outgoing flow
                for j in self.nodes:
                    if (i,j) in self.arcs:
                        expr -= w_vars[v, (i,j)]
                
                # Incoming flow  
                for j in self.nodes:
                    if (j,i) in self.arcs:
                        expr += w_vars[v, (j,i)]
                
                # Right-hand side
                rhs = 0
                if i == v:  # Source node
                    total_demand = sum(self.demands[f] for f in self.commodities_by_origin[v])
                    total_is = sum(is_vars[f] for f in self.commodities_by_origin[v])
                    rhs = total_is - total_demand
                else:  # Other nodes
                    for f in self.commodities_by_origin[v]:
                        if i == f[1]:  # Destination of commodity f
                            rhs = self.demands[f] - is_vars[f]
                
                model.addConstr(expr == rhs, name=f"flow_cons_{v}_{i}")
        
        # Optimize
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            return model, u_max.X, {k: v.X for k,v in w_vars.items()}, {k: v.X for k,v in is_vars.items()}
        else:
            return None, None, None, None
    
    def visualize_solution(self, w_values, is_values):
        """Visualize the network and routing solution"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.G, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1)
        
        # Show capacities
        edge_labels = {(i,j): f"{self.capacities.get((i,j), 0)}" 
                      for i,j in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)
        
        plt.title("Network Topology with Link Capacities")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def create_test_instance():
    """Create the small test instance from the paper (Figure 2)"""
    
    # Create graph
    G = nx.Graph()
    nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    G.add_nodes_from(nodes)
    
    # Add edges with capacities (all 1000 Mbps as in paper)
    edges = [('a', 'c'), ('c', 'f'), ('f', 'h'), ('h', 'e'), 
             ('e', 'g'), ('g', 'd'), ('d', 'b'), ('b', 'a'),
             ('c', 'd'), ('f', 'g')]
    
    capacities = {}
    for i, j in edges:
        G.add_edge(i, j)
        capacities[(i,j)] = 1000
        capacities[(j,i)] = 1000
    
    # Traffic matrix from Figure 2
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
    
    # Compute shortest path routing (unit weights)
    routing_matrix = compute_ospf_routing(G, traffic_matrix)
    
    return G, capacities, traffic_matrix, routing_matrix

def compute_ospf_routing(G, demands):
    """Compute OSPF routing matrix using shortest paths"""
    routing_matrix = {}
    
    for (s, t), demand in demands.items():
        # Find shortest path
        try:
            path = nx.shortest_path(G, s, t)
            
            # Set routing fractions
            for i in range(len(path) - 1):
                arc = (path[i], path[i+1])
                routing_matrix[(s,t), arc] = 1.0
        except nx.NetworkXNoPath:
            pass
    
    return routing_matrix

def main():
    print("=== Combined IGP/MPLS-TE Routing Optimization ===")
    print("Creating test instance...")
    
    # Create test instance
    G, capacities, demands, routing_matrix = create_test_instance()
    
    print(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Commodities: {len(demands)}")
    print(f"Total demand: {sum(demands.values())} Mbps")
    
    # Create optimizer
    optimizer = CombinedRoutingOptimizer(G, capacities, demands, routing_matrix)
    
    # First compute IGP-only utilization for comparison
    print("\nComputing IGP-only utilization...")
    igp_loads = {}
    for arc in optimizer.arcs:
        load = 0
        for f in optimizer.commodities:
            if (f, arc) in routing_matrix:
                load += demands[f] * routing_matrix[f, arc]
        igp_loads[arc] = load
    
    max_igp_util = 0
    for arc in optimizer.arcs:
        if arc in capacities and capacities[arc] > 0:
            util = igp_loads[arc] / capacities[arc]
            max_igp_util = max(max_igp_util, util)
    
    print(f"IGP-only max utilization: {max_igp_util * 100:.1f}%")
    
    # Solve nominal problem
    print("\nSolving nominal routing problem...")
    model, u_max, w_values, is_values = optimizer.solve_nominal(delta=0.000001)  # Much smaller delta
    
    if model:
        print(f"\nOptimal solution found!")
        print(f"Maximum link utilization: {u_max * 100:.1f}%")  # Convert to percentage
        
        # Count LSPs
        lsp_count = sum(1 for v,arc in w_values.keys() 
                       if arc[0] == v and w_values[v,arc] > 0.01)
        print(f"Number of LSPs: {lsp_count}")
        
        # Show traffic split
        total_igp = sum(is_values.values())
        total_demand = sum(demands.values())
        print(f"IGP traffic: {total_igp:.1f} Mbps ({100*total_igp/total_demand:.1f}%)")
        print(f"MPLS traffic: {total_demand - total_igp:.1f} Mbps ({100*(1-total_igp/total_demand):.1f}%)")
        
        # Visualize
        optimizer.visualize_solution(w_values, is_values)
        
        # Show some LSP details
        print("\nTop 5 LSPs by traffic volume:")
        lsp_list = []
        for v in optimizer.origin_nodes:
            for arc in optimizer.arcs:
                if arc[0] == v and w_values[v,arc] > 0.01:
                    lsp_list.append((v, arc[1], w_values[v,arc]))
        
        lsp_list.sort(key=lambda x: x[2], reverse=True)
        for i, (src, next_hop, flow) in enumerate(lsp_list[:5]):
            print(f"  LSP from {src} via {next_hop}: {flow:.1f} Mbps")
    else:
        print("No optimal solution found!")

if __name__ == "__main__":
    main()