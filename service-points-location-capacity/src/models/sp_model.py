"""
Main optimization model for Service Points Location and Capacity Problem
Based on Raviv (2023) - Transportation Research Part E
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Tuple, Optional
from .rejection_function import RejectionFunction
import os

class ServicePointModel:
    """
    MILP model for optimal location and capacity of service points
    Following formulation from Raviv (2023)
    """
    
    def __init__(self, 
                 demand_points: List[Tuple[float, float]], 
                 candidate_locations: List[Tuple[float, float]],
                 capacities: List[int],
                 demand_rates: List[float],
                 params: Dict):
        """
        Initialize the model
        
        Parameters:
        -----------
        demand_points : List of (x, y) coordinates
        candidate_locations : List of (x, y) coordinates for potential SPs  
        capacities : List of possible capacities
        demand_rates : List of demand rates μ_d for each demand point
        params : Dictionary with model parameters
        """
        self.D = demand_points
        self.F = candidate_locations
        self.C = capacities
        self.mu = demand_rates
        self.params = params
        
        # Extract parameters
        self.r = params.get('service_radius', 600)
        self.p = params.get('pickup_probability', 0.5)
        self.d = params.get('rejection_cost', 10)  # α in the paper turns in d
        self.setup_base_cost = params.get('setup_base_cost', 10)
        self.setup_var_cost = params.get('setup_var_cost', 1/6)
        
        # Calculate distances
        self.distances = self._calculate_distances()
        
        # Precompute rejection functions for each capacity
        self.rejection_functions = {
            c: RejectionFunction(c, self.p) for c in self.C
        }
        
        # Piecewise linear approximation points
        self.n_breakpoints = params.get('n_breakpoints', 12)
        self._prepare_pwl_data()
        
        # Initialize Gurobi model
        self.model = gp.Model("SP_Location_Capacity")
        
        # Decision variables (to be created)
        self.y = None  # Binary variables: SP locations/capacities
        self.x = None  # Flow variables: demand allocation
        self.z = None  # PWL approximation variables
        
    def _calculate_distances(self) -> Dict[Tuple[int, int], float]:
        """Calculate Euclidean distances between all points"""
        distances = {}
        for i, d in enumerate(self.D):
            for j, f in enumerate(self.F):
                dist = np.sqrt((d[0] - f[0])**2 + (d[1] - f[1])**2)
                distances[(i, j)] = dist
        return distances
    
    def _prepare_pwl_data(self):    
        """Prepare piecewise linear approximation data"""
        print("Preparing PWL approximation data...")
    
        self.pwl_data = {}
    
        for s_idx, capacity in enumerate(self.C):
            rf = RejectionFunction(capacity, self.p)
        
            # Find maximum lambda for each location
            max_lambda_f = {}
            for f in range(len(self.F)):
                max_lambda_f[f] = sum(
                    self.mu[d] for d in range(len(self.D))
                    if self.distances[(d, f)] <= self.r
                )
        
            max_lambda = max(max_lambda_f.values()) if max_lambda_f else 100
        
            # Get PWL points
            pwl_points = rf.get_piecewise_linear_points(
                max_lambda * 1.2,  # 20% margin
                n_points=self.n_breakpoints
            )
        
            # Progress tracking
            print(f"  Capacity {capacity}: {len(pwl_points['lambda_values'])} breakpoints generated")
        
            # Store data for each breakpoint
            for k_idx in range(self.n_breakpoints):
                if k_idx < len(pwl_points['lambda_values']):
                    self.pwl_data[(s_idx, k_idx)] = {
                        "lambda_value": pwl_points['lambda_values'][k_idx],
                        "rejection_value": pwl_points['rejection_values'][k_idx]
                    }
                else:
                    # Extend with the last value if there are no breakpoints
                    self.pwl_data[(s_idx, k_idx)] = {
                        "lambda_value": pwl_points['lambda_values'][-1] + (k_idx - len(pwl_points['lambda_values']) + 1) * 10,
                        "rejection_value": pwl_points['rejection_values'][-1]
                    }
    
        # Summary
        print(f"PWL data keys created: {len(self.pwl_data)} total")
    
    def _get_setup_cost(self, f: int, s: int) -> float:
        """Calculate setup cost for location f with capacity s
        Paper uses base=10k$ meaning 10,000$ base cost
        """
        base = self.setup_base_cost * 1000  # Convert to thousands
        variable = self.setup_var_cost * self.C[s] * 1000  # Convert to thousands
        location_factor = 1.0  # Location-specific factor (currently uniform)
        return (base + variable) * location_factor
    
    def build_model(self):
        """Build the complete MILP model following Raviv (2023)"""
        print("Building MILP model...")
        
        # Index sets
        D_idx = range(len(self.D))
        F_idx = range(len(self.F))
        S_idx = range(len(self.C))
        K_idx = range(self.n_breakpoints)
        
        # Decision variables
        print("Creating decision variables...")
        
        # y[f,s] = 1 if SP with capacity s opened at location f
        self.y = self.model.addVars(
            F_idx, S_idx, 
            vtype=GRB.BINARY, 
            name="y"
        )
        
        # x[d,f] = flow from demand point d to SP f
        self.x = self.model.addVars(
            [(d, f) for d in D_idx for f in F_idx 
             if self.distances[(d, f)] <= self.r],
            lb=0, 
            vtype=GRB.CONTINUOUS, 
            name="x"
        )
        
        # z[f,s,k] for piecewise linear approximation
        self.z = self.model.addVars(
            F_idx, S_idx, K_idx,
            lb=0, 
            ub=1,
            vtype=GRB.CONTINUOUS, 
            name="z"
        )
        
        # Objective function (8)
        print("Setting objective function...")
        obj = self._build_objective(F_idx, S_idx, K_idx)
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        print("Adding constraints...")
        self._add_constraints(D_idx, F_idx, S_idx, K_idx)
        
        self.model.update()
        
        print("Model building complete!")
        print(f"\nModel statistics:")
        print(f"  Variables: {self.model.NumVars}")
        print(f"  Constraints: {self.model.NumConstrs}")
        print(f"  Non-zeros: {self.model.NumNZs}")
        
    def _build_objective(self, F_idx, S_idx, K_idx):
        """Build objective function (8) from the paper"""
        # Setup costs - first term(8)
        setup_cost = gp.quicksum(
            self._get_setup_cost(f, s) * self.y[f, s]
            for f in F_idx for s in S_idx
        )
        
        # Rejection costs - second term (8)
    
        rejection_cost = gp.quicksum(
            self.d * self.pwl_data[(s, k)]["rejection_value"] * self.z[f, s, k]
            for f in F_idx for s in S_idx for k in K_idx
        )
        
        return setup_cost + rejection_cost
    
    def _add_constraints(self, D_idx, F_idx, S_idx, K_idx):
        """Add all model constraints from the paper"""
        uncovered_count = 0

        # (9) At most one capacity per location
        for f in F_idx:
            self.model.addConstr(
                gp.quicksum(self.y[f, s] for s in S_idx) <= 1,
                name=f"one_size_per_location_{f}"
            )
        
        # (10) Satisfy all demand
        for d in D_idx:
            eligible_sps = [f for f in F_idx if self.distances[(d, f)] <= self.r]
            if not eligible_sps:
                uncovered_count += 1
                if uncovered_count <= 10: 
                    print(f"Uncovered demand point {d} (no SP within {self.r}m)")
                
                slack = self.model.addVar(lb=0, name=f"slack_{d}")
                self.model.addConstr(
                    slack == self.mu[d],
                    name=f"uncovered_demand_{d}"
                )
                # Penalize object function
                self.model.setObjective(
                    self.model.getObjective() + 1000000 * slack,
                    GRB.MINIMIZE
                )
            else:
                self.model.addConstr(
                    gp.quicksum(self.x[d, f] for f in eligible_sps) == self.mu[d],
                    name=f"satisfy_demand_{d}"
                )
    
        if uncovered_count > 10:
            print(f"... and others {uncovered_count - 10} not covered points")
    
        if uncovered_count > 0:
            print(f"\nTOTAL of not covered points: {uncovered_count}/{len(D_idx)} ({uncovered_count/len(D_idx)*100:.1f}%)")
            print("Model includes penalty terms for uncovered points")
        
        # (11) Closest SP constraint
        for d in D_idx:
            for f in F_idx:
                if self.distances[(d, f)] <= self.r:
                    # Find SPs farther than f
                    farther_sps = [
                        fp for fp in F_idx 
                        if self.distances[(d, fp)] > self.distances[(d, f)]
                        and self.distances[(d, fp)] <= self.r
                    ]
                    
                    if farther_sps:
                        self.model.addConstr(
                            gp.quicksum(self.x[d, fp] for fp in farther_sps) <= 
                            self.mu[d] * (1 - gp.quicksum(self.y[f, s] for s in S_idx)),
                            name=f"closest_sp_{d}_{f}"
                        )
        
        # (14) PWL approximation - lambda representation
        for f in F_idx:
            # Arrival rate at SP f
            self.model.addConstr(
                gp.quicksum(self.x[d, f] for d in D_idx if (d, f) in self.x) ==
                gp.quicksum(
                    gp.quicksum(
                        self.pwl_data[(s, k)]["lambda_value"] * self.z[f, s, k]
                        for k in K_idx
                    ) for s in S_idx
                ),
                name=f"pwl_lambda_{f}"
            )
        
        # (15) Convex combination constraint
        for f in F_idx:
            for s in S_idx:
                self.model.addConstr(
                    gp.quicksum(self.z[f, s, k] for k in K_idx) == self.y[f, s],
                    name=f"convex_combination_{f}_{s}"
                )
    
    def solve(self, time_limit: int = 3600, mip_gap: float = 0.01) -> Dict:
        """
        Solve the model
        
        Parameters:
        -----------
        time_limit : int
            Time limit in seconds
        mip_gap : float
            MIP optimality gap
            
        Returns:
        --------
        dict
            Solution details
        """
        print(f"\nSolving model (time limit: {time_limit}s, gap: {mip_gap})...")
        
        # Set parameters
        self.model.Params.TimeLimit = time_limit
        self.model.Params.MIPGap = mip_gap
        self.model.Params.OutputFlag = 1  # Enable output
        
        # Solve
        self.model.optimize()
        
        # Extract solution
        if self.model.Status == GRB.OPTIMAL:
            print("Optimal solution found!")
        elif self.model.Status == GRB.TIME_LIMIT:
            print("Time limit reached.")
        elif self.model.Status == GRB.INFEASIBLE:
            print("Model is infeasible!")
            self.model.computeIIS()
            return {"status": "infeasible"}
        
        return self._extract_solution()
    
    def _extract_solution(self) -> Dict:
        """Extract and format solution"""
        solution = {
            "status": self.model.Status,
            "objective_value": self.model.ObjVal if self.model.SolCount > 0 else None,
            "mip_gap": self.model.MIPGap if self.model.SolCount > 0 else None,
            "runtime": self.model.Runtime,
            "node_count": self.model.NodeCount,
            "service_points": [],
            "allocations": [],
            "summary": {}
        }
        
        if self.model.SolCount == 0:
            return solution
        
        # Extract opened SPs
        total_capacity = 0
        sp_flows = {}  # Track flow to each SP
        
        for f in range(len(self.F)):
            for s in range(len(self.C)):
                if self.y[f, s].X > 0.5:
                    sp_info = {
                        "location_idx": f,
                        "location": self.F[f],
                        "capacity": self.C[s],
                        "setup_cost": self._get_setup_cost(f, s),
                        "arrival_rate": 0.0,  # Will be calculated
                        "utilization": 0.0,   # Will be calculated
                        "expected_rejections": 0.0  # Will be calculated
                    }
                    solution["service_points"].append(sp_info)
                    total_capacity += self.C[s]
                    sp_flows[f] = 0.0
        
        # Extract allocations and calculate flows
        total_flow = 0
        for (d, f), var in self.x.items():
            if var.X > 0.001:
                solution["allocations"].append({
                    "demand_point": d,
                    "service_point": f,
                    "flow": var.X,
                    "distance": self.distances[(d, f)]
                })
                total_flow += var.X
                
                # Track flow to SP
                if f in sp_flows:
                    sp_flows[f] += var.X
        
        # Calculate actual utilization and rejections for each SP
        total_rejections = 0
        for sp in solution["service_points"]:
            f = sp["location_idx"]
            arrival_rate = sp_flows.get(f, 0.0)
            sp["arrival_rate"] = arrival_rate
            
            # Utilization = arrival_rate / (capacity * pickup_probability)
            effective_capacity = sp["capacity"] * self.p
            sp["utilization"] = arrival_rate / effective_capacity if effective_capacity > 0 else 0
            
            # Calculate expected rejections using the rejection function
            rf = self.rejection_functions[sp["capacity"]]
            sp["expected_rejections"] = rf.expected_rejections(arrival_rate)
            total_rejections += sp["expected_rejections"]
            
            # Add warning if utilization > 100%
            if sp["utilization"] > 1.0:
                print(f"SP at {sp['location']}: utilization {sp['utilization']:.1%} (overloaded)")
                print(f"  Arrival rate: {arrival_rate:.2f}, Effective capacity: {effective_capacity:.2f}")
        
        # Summary statistics
        avg_utilization = sum(sp["utilization"] * sp["capacity"] for sp in solution["service_points"]) / total_capacity if total_capacity > 0 else 0
        
        solution["summary"] = {
            "num_service_points": len(solution["service_points"]),
            "total_capacity": total_capacity,
            "total_demand": sum(self.mu),
            "total_flow": total_flow,
            "avg_utilization": avg_utilization,
            "max_utilization": max(sp["utilization"] for sp in solution["service_points"]) if solution["service_points"] else 0,
            "total_setup_cost": sum(sp["setup_cost"] for sp in solution["service_points"]),
            "total_rejection_cost": self.d * total_rejections,
            "expected_total_rejections": total_rejections
        }
        
        # Verify solution feasibility
        if solution["summary"]["max_utilization"] > 1.0:
            print(f"\nOverloaded SPs detected (max utilization: {solution['summary']['max_utilization']:.1%})")
            print("Rejections will handle overflow as per Raviv's model")
        
        # Diagnostica rejections
        print(f"\nDiagnostica rejections:")
        for sp in solution["service_points"]:
            print(f"SP at {sp['location']}:")
            print(f"  Capacity: {sp['capacity']}")
            print(f"  Arrival rate: {sp['arrival_rate']:.2f}")
            print(f"  Load ratio rho: {sp['arrival_rate']/(sp['capacity']*self.p):.2f}")
            print(f"  Expected rejections: {sp['expected_rejections']:.2f}")
            
        return solution