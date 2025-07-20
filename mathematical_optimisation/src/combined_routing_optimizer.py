import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from path_decomposer import PathDecomposer
from ecmp_handler import ECMPHandler

class CombinedRoutingOptimizer:
    def __init__(self, graph, capacities, demands, routing_matrix, use_ecmp=True):
        """
        Initialize the optimizer
        
        Parameters:
        - graph: NetworkX graph representing the network topology
        - capacities: Dictionary {(i,j): capacity} for each link
        - demands: Dictionary {(s,t): demand} for each commodity
        - routing_matrix: IGP routing matrix X where X[f,(i,j)] is fraction of flow f on arc (i,j)
        - use_ecmp: Whether to use ECMP routing
        """
        self.G = graph
        self.capacities = capacities
        self.demands = demands
        self.routing_matrix = routing_matrix
        self.use_ecmp = use_ecmp
        
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
        
        # Compute new shortest paths (with ECMP if enabled)
        failure_routing_matrix = {}
        
        if self.use_ecmp:
            temp_demands = {(s, t): 1 for s, t in self.demands.keys()}
            failure_routing_matrix = ECMPHandler.compute_ecmp_routing(G_temp, temp_demands)
        else:
            for (s, t), demand in self.demands.items():
                try:
                    path = nx.shortest_path(G_temp, s, t)
                    for i in range(len(path) - 1):
                        arc = (path[i], path[i+1])
                        failure_routing_matrix[(s,t), arc] = 1.0
                except nx.NetworkXNoPath:
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
            
            # Path decomposition
            paths = PathDecomposer.decompose_flow(self.G, result['w_values'], self.origin_nodes)
            result['lsp_paths'] = paths
            result['lsp_count'] = len(paths)
            
            # Store top LSPs for visualization - FIX: store full paths, not just first hop
            result['lsp_details'] = [(p[0], p[1], p[3]) for p in sorted(paths, 
                                     key=lambda x: x[3], reverse=True)[:10]]
            # Store full paths for visualization
            result['lsp_full_paths'] = [(p[0], p[1], p[2], p[3]) for p in sorted(paths, 
                                        key=lambda x: x[3], reverse=True)[:10]]
            
            return result
        else:
            return {'status': 'infeasible'}
    
    def solve_with_survivability(self, failure_scenarios=None, delta=0, verbose=False):
        """
        Solve with survivability constraints using Equation 7 from the paper
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
            # Extract results (similar to nominal case)
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
            
            # Path decomposition
            paths = PathDecomposer.decompose_flow(self.G, result['w_values'], self.origin_nodes)
            result['lsp_paths'] = paths
            result['lsp_count'] = len(paths)
            result['lsp_details'] = [(p[0], p[1], p[3]) for p in sorted(paths, 
                                     key=lambda x: x[3], reverse=True)[:10]]
            # Store full paths for visualization
            result['lsp_full_paths'] = [(p[0], p[1], p[2], p[3]) for p in sorted(paths, 
                                        key=lambda x: x[3], reverse=True)[:10]]
            
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
        
        # Show path details if available
        if 'lsp_paths' in result and result['lsp_paths']:
            print(f"\n  Top LSP Paths:")
            for i, (src, dst, path, flow) in enumerate(result['lsp_paths'][:5]):
                path_str = " -> ".join(path)
                print(f"    {i+1}. {src} to {dst}: {flow:.1f} Mbps")
                print(f"       Path: {path_str}")