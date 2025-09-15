"""
Deterministic benchmark model for comparison
Safety stock approach without stochasticity
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Tuple


class DeterministicModel:
    """
    Deterministic model that uses safety stock approach
    """
    
    def __init__(self, 
                 demand_points: List[Tuple[float, float]], 
                 candidate_locations: List[Tuple[float, float]],
                 capacities: List[int],
                 demand_rates: List[float],
                 params: Dict,
                 safety_factor: float = 1.2):
        """
        Initialize deterministic model
        
        Parameters:
        -----------
        safety_factor : float
            Multiplier for demand to account for uncertainty (>1)
        """
        self.D = demand_points
        self.F = candidate_locations
        self.C = capacities
        self.mu = demand_rates
        self.params = params
        self.safety_factor = safety_factor
        
        # Extract parameters
        self.r = params.get('service_radius', 600)
        self.p = params.get('pickup_probability', 0.5)
        self.setup_base_cost = params.get('setup_base_cost', 10)
        self.setup_var_cost = params.get('setup_var_cost', 1/6)
        
        # Calculate distances
        self.distances = self._calculate_distances()
        
        # Initialize model
        self.model = gp.Model("Deterministic_SP")
        self.model_type = "deterministic"
        
    def _calculate_distances(self) -> Dict[Tuple[int, int], float]:
        """Calculate Euclidean distances"""
        distances = {}
        for i, d in enumerate(self.D):
            for j, f in enumerate(self.F):
                dist = np.sqrt((d[0] - f[0])**2 + (d[1] - f[1])**2)
                distances[(i, j)] = dist
        return distances
    
    def _get_setup_cost(self, f: int, s: int) -> float:
        """Calculate setup cost
        Paper uses base=10k$ meaning 10,000$ base cost
        """
        base = self.setup_base_cost * 1000  # Convert to thousands
        variable = self.setup_var_cost * self.C[s] * 1000  # Convert to thousands
        return base + variable
    
    def build_model(self):
        """Build deterministic MILP model"""
        print(f"Building DETERMINISTIC model (safety factor={self.safety_factor})...")
        
        # Index sets
        D_idx = range(len(self.D))
        F_idx = range(len(self.F))
        S_idx = range(len(self.C))
        
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
        
        
        self.unmet = self.model.addVars(
            D_idx,
            lb=0,
            vtype=GRB.CONTINUOUS,
            name="unmet_demand"
        )
        
        # Objective: setup costs + rejection cost (same as stochastic model)
        print("Setting objective (setup costs + rejection cost)...")
        
        # Use same rejection cost as stochastic model (α)
        rejection_cost = self.params.get('rejection_cost', 10)
        
        obj = gp.quicksum(
            self._get_setup_cost(f, s) * self.y[f, s]
            for f in F_idx for s in S_idx
        ) + rejection_cost * gp.quicksum(self.unmet[d] for d in D_idx)
        
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        print("Adding constraints...")
        
        # At most one capacity per location
        for f in F_idx:
            self.model.addConstr(
                gp.quicksum(self.y[f, s] for s in S_idx) <= 1,
                name=f"one_size_{f}"
            )
        
        # Satisfy demand (con slack per feasibility)
        for d in D_idx:
            eligible_sps = [f for f in F_idx if self.distances[(d, f)] <= self.r]
            if eligible_sps:
                self.model.addConstr(
                    gp.quicksum(self.x[d, f] for f in eligible_sps) + self.unmet[d] == self.mu[d],
                    name=f"satisfy_demand_{d}"
                )
            else:
                
                self.model.addConstr(
                    self.unmet[d] == self.mu[d],
                    name=f"no_coverage_{d}"
                )
        
        # Capacity constraints WITH SAFETY FACTOR
        # The safety factor reduces the effective capacity to account for uncertainty
        for f in F_idx:
            arrival = gp.quicksum(
                self.x[d, f] for d in D_idx if (d, f) in self.x
            )
            
            # Reduce effective capacity by safety factor
            effective_capacity = gp.quicksum(
                self.C[s] * self.p * self.y[f, s] / self.safety_factor
                for s in S_idx
            )
            
            self.model.addConstr(
                arrival <= effective_capacity,
                name=f"capacity_{f}"
            )
        
        # Flow only if SP is open
        for d in D_idx:
            for f in F_idx:
                if (d, f) in self.x:
                    self.model.addConstr(
                        self.x[d, f] <= self.mu[d] * gp.quicksum(self.y[f, s] for s in S_idx),
                        name=f"flow_if_open_{d}_{f}"
                    )
       
        
        self.model.update()
        print(f"\nDeterministic model statistics:")
        print(f"  Variables: {self.model.NumVars}")
        print(f"  Constraints: {self.model.NumConstrs}")
        print(f"  Safety factor: {self.safety_factor}")
    
    def solve(self, time_limit: int = 3600, mip_gap: float = 0.01) -> Dict:
        """Solve the model"""
        print(f"\nSolving model (time limit: {time_limit}s, gap: {mip_gap})...")
        
        # Set parameters
        self.model.Params.TimeLimit = time_limit
        self.model.Params.MIPGap = mip_gap
        self.model.Params.OutputFlag = 1
        
        # Solve
        self.model.optimize()
        
        # Extract solution
        if self.model.Status == GRB.OPTIMAL:
            print("Optimal solution found!")
        elif self.model.Status == GRB.TIME_LIMIT:
            print("Time limit reached.")
        elif self.model.Status == GRB.INFEASIBLE:
            print("Model is infeasible!")
            
            return {"status": "infeasible"}
        
        return self._extract_solution()
    
    def _extract_solution(self) -> Dict:
        """Extract solution details"""
        solution = {
            "status": self.model.Status,
            "objective_value": self.model.ObjVal if self.model.SolCount > 0 else None,
            "mip_gap": self.model.MIPGap if self.model.SolCount > 0 else None,
            "runtime": self.model.Runtime,
            "node_count": self.model.NodeCount,
            "service_points": [],
            "summary": {}
        }
        
        if self.model.SolCount == 0:
            return solution
        
        # Control not satisfied demand
        total_unmet = sum(self.unmet[d].X for d in range(len(self.D)))
        if total_unmet > 0.001:
            print(f"\nATTENZIONE: Domanda non soddisfatta = {total_unmet:.2f}")
        
        # Extract opened SPs
        total_capacity = 0
        sp_flows = {}
        
        for f in range(len(self.F)):
            for s in range(len(self.C)):
                if self.y[f, s].X > 0.5:
                    sp_info = {
                        "location_idx": f,
                        "location": self.F[f],
                        "capacity": self.C[s],
                        "setup_cost": self._get_setup_cost(f, s),
                        "arrival_rate": 0.0,
                        "utilization": 0.0
                    }
                    solution["service_points"].append(sp_info)
                    total_capacity += self.C[s]
                    sp_flows[f] = 0.0
        
        # Calculate flows
        total_flow = 0
        for (d, f), var in self.x.items():
            if var.X > 0.001:
                total_flow += var.X
                if f in sp_flows:
                    sp_flows[f] += var.X
        
        # Update SP info
        for sp in solution["service_points"]:
            f = sp["location_idx"]
            arrival_rate = sp_flows.get(f, 0.0)
            sp["arrival_rate"] = arrival_rate
            
            # utilization with safety factor applied to capacity
            effective_capacity = sp["capacity"] * self.p / self.safety_factor
            sp["utilization"] = arrival_rate / effective_capacity if effective_capacity > 0 else 0
        
        # Summary
        avg_utilization = sum(sp["utilization"] * sp["capacity"] for sp in solution["service_points"]) / total_capacity if total_capacity > 0 else 0
        
        # Calculate costs properly 
        actual_setup_cost = sum(sp["setup_cost"] for sp in solution["service_points"])
        rejection_cost = self.params.get('rejection_cost', 10)
        actual_rejection_cost = total_unmet * rejection_cost
        
        solution["summary"] = {
            "num_service_points": len(solution["service_points"]),
            "total_capacity": total_capacity,
            "total_demand": sum(self.mu),
            "total_flow": total_flow,
            "unmet_demand": total_unmet,
            "avg_utilization": avg_utilization,
            "max_utilization": max(sp["utilization"] for sp in solution["service_points"]) if solution["service_points"] else 0,
            "total_setup_cost": actual_setup_cost,
            "total_rejection_cost": actual_rejection_cost,
            "expected_total_rejections": total_unmet
        }
        
   
        
        return solution