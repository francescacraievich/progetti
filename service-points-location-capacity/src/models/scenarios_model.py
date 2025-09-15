"""
Scenario-based stochastic model from Raviv (2023) Section 5.1
Uses multiple demand scenarios
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from .sp_model import ServicePointModel


class ScenariosModel(ServicePointModel):
    """
    Scenario-based model with multiple demand scenarios
    """
    
    def __init__(self, *args, n_scenarios: int = 30, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "scenarios"
        self.n_scenarios = n_scenarios
        self._generate_scenarios()
    
    def _generate_scenarios(self):
        """Generate demand scenarios using normal distribution"""
        self.scenarios = []
        np.random.seed(42)  # For reproducibility
        
        for s in range(self.n_scenarios):
            scenario = []
            for mu_d in self.mu:
                # Normal distribution with CV = 0.33 (std = mean/3)
                demand = max(0, np.random.normal(mu_d, mu_d/3))
                scenario.append(demand)
            self.scenarios.append(scenario)
    
    def build_model(self):
        """Build scenario-based stochastic model"""
        print(f"Building SCENARIOS model with {self.n_scenarios} scenarios...")
        
        # Index sets
        D_idx = range(len(self.D))
        F_idx = range(len(self.F))
        S_idx = range(len(self.C))
        Sc_idx = range(self.n_scenarios)
        K_idx = range(self.n_breakpoints)  
        
        # First-stage: Location decisions
        self.y = self.model.addVars(
            F_idx, S_idx, 
            vtype=GRB.BINARY, 
            name="y"
        )
        
        # Second-stage: Flow and rejection decisions per scenario
        self.x = {}
        self.z = {}
        
        for sc in Sc_idx:
            # Flow variables per scenario
            self.x[sc] = self.model.addVars(
                [(d, f) for d in D_idx for f in F_idx 
                 if self.distances[(d, f)] <= self.r],
                lb=0, 
                vtype=GRB.CONTINUOUS, 
                name=f"x_sc{sc}"
            )
            
            # PWL variables for rejection function per scenario
            self.z[sc] = self.model.addVars(
                F_idx, S_idx, K_idx, 
                lb=0, ub=1,
                vtype=GRB.CONTINUOUS,
                name=f"z_sc{sc}"
            )
        
        # Objective: Setup + Expected rejection costs
        print("Setting objective function...")
        
        setup_cost = gp.quicksum(
            self._get_setup_cost(f, s) * self.y[f, s]
            for f in F_idx for s in S_idx
        )
        
        # Expected rejection cost over all scenarios
        expected_rejection_cost = 0
        for sc in Sc_idx:
            for f in F_idx:
                for s in S_idx:
                    for k in K_idx:
                        
                        expected_rejection_cost += (
                            self.d * self.pwl_data[(s, k)]["rejection_value"] * 
                            self.z[sc][f, s, k] / self.n_scenarios
                        )
        
        self.model.setObjective(setup_cost + expected_rejection_cost, GRB.MINIMIZE)
        
        # Constraints
        print("Adding constraints...")
        
        # First-stage constraints
        for f in F_idx:
            self.model.addConstr(
                gp.quicksum(self.y[f, s] for s in S_idx) <= 1,
                name=f"one_size_{f}"
            )
        
        # Second-stage constraints per scenario
        for sc in Sc_idx:
            # Demand satisfaction
            for d in D_idx:
                eligible_sps = [f for f in F_idx if self.distances[(d, f)] <= self.r]
                if eligible_sps:
                    self.model.addConstr(
                        gp.quicksum(self.x[sc][d, f] for f in eligible_sps) == 
                        self.scenarios[sc][d],
                        name=f"demand_sc{sc}_{d}"
                    )
            
            # PWL constraints
            for f in F_idx:
                # Arrival rate
                self.model.addConstr(
                    gp.quicksum(self.x[sc][d, f] for d in D_idx if (d, f) in self.x[sc]) ==
                    gp.quicksum(
                        gp.quicksum(
                            self.pwl_data[(s, k)]["lambda_value"] * self.z[sc][f, s, k]
                            for k in K_idx
                        ) for s in S_idx
                    ),
                    name=f"pwl_arrival_sc{sc}_{f}"
                )
                
                # PWL convexity
                for s in S_idx:
                    self.model.addConstr(
                        gp.quicksum(self.z[sc][f, s, k] for k in K_idx) == self.y[f, s],
                        name=f"pwl_convex_sc{sc}_{f}_{s}"
                    )
            
            # Closest SP constraint
            for d in D_idx:
                for f in F_idx:
                    if self.distances[(d, f)] <= self.r:
                        farther_sps = [
                            fp for fp in F_idx 
                            if self.distances[(d, fp)] > self.distances[(d, f)]
                            and self.distances[(d, fp)] <= self.r
                        ]
                        if farther_sps:
                            self.model.addConstr(
                                gp.quicksum(self.x[sc][d, fp] for fp in farther_sps) <= 
                                self.scenarios[sc][d] * (1 - gp.quicksum(self.y[f, s] for s in S_idx)),
                                name=f"closest_sc{sc}_{d}_{f}"
                            )
        
        self.model.update()
        print(f"Scenarios model building complete!")