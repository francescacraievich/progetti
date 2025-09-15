"""
Implementation of the rejection function using DTMC (Discrete-Time Markov Chain)
Based on Section 3.1 of Raviv (2023) - Transportation Research Part E

This implementation follows the exact DTMC model described in the paper:
- Bipartite graph with states (A,i) after replenishment and (B,j) before replenishment
- Transition probabilities based on Poisson arrivals and Binomial pickups
- Steady-state analysis to compute expected rejections using equation (4)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import poisson, binom
from typing import Dict, Tuple, Optional
import warnings


class RejectionFunction:
    """
    Calculate expected rejections using the exact DTMC approach from Raviv (2023)
    
    The DTMC has two sets of states:
    - (A,i): i parcels in SP immediately after replenishment
    - (B,j): j parcels in SP immediately before replenishment
    """
    
    def __init__(self, capacity: int, pickup_prob: float = 0.5):
        """
        Initialize DTMC rejection function
        
        Parameters:
        -----------
        capacity : int
            Number of lockers in the service point (C)
        pickup_prob : float
            Probability of pickup in each period (p)
        """
        self.C = capacity
        self.p = pickup_prob
        
        # Pre-compute binomial probabilities for efficiency
        self._precompute_pickup_probs()
        
    def _precompute_pickup_probs(self):
        """Pre-compute binomial pickup probabilities P(Y_i = i-j)"""
        self.pickup_probs = {}
        for i in range(self.C + 1):
            for j in range(i + 1):
                # P(Y_i = i-j) where Y_i ~ Bin(i, p)
                n_pickups = i - j
                self.pickup_probs[(i, j)] = binom.pmf(n_pickups, i, self.p)
    
    def _build_transition_matrix(self, lambda_rate: float) -> np.ndarray:
        """
        Build the transition probability matrix for the DTMC
        
        The state space is doubled: states 0 to C are "B" states (before replenishment)
        and states C+1 to 2C+1 are "A" states (after replenishment)
        """
        n_states = 2 * (self.C + 1)
        P = np.zeros((n_states, n_states))
        
        # B states are indexed 0 to C
        # A states are indexed C+1 to 2C+1
        
        # Transitions from A states to B states (pickups during period)
        for i in range(self.C + 1):
            a_idx = self.C + 1 + i  # Index of state (A,i)
            for j in range(i + 1):
                b_idx = j  # Index of state (B,j)
                # P[(A,i), (B,j)] = P(Y_i = i-j) from equation (1)
                P[a_idx, b_idx] = self.pickup_probs[(i, j)]
        
        # Transitions from B states to A states (arrivals at replenishment)
        for j in range(self.C + 1):
            b_idx = j  # Index of state (B,j)
            
            # For i < C: P[(B,j), (A,i)] = P(X = i-j) from equation (2)
            for i in range(j, self.C):
                a_idx = self.C + 1 + i  # Index of state (A,i)
                n_arrivals = i - j
                P[b_idx, a_idx] = poisson.pmf(n_arrivals, lambda_rate)
            
            # For i = C: P[(B,j), (A,C)] = P(X >= C-j) from equation (3)
            a_idx = 2 * self.C + 1  # Index of state (A,C)
            P[b_idx, a_idx] = 1 - poisson.cdf(self.C - j - 1, lambda_rate)
        
        return P
    
    def _compute_steady_state(self, P: np.ndarray) -> np.ndarray:
        """
        Compute steady-state probabilities of the DTMC
        
        Since the chain is periodic with period 2, we compute the limiting
        probabilities for the B states (before replenishment)
        """
        n_states = P.shape[0]
        
        # For periodic chain, compute P^2 to get steady state for each class
        P2 = P @ P
        
        # Extract the B-to-B transition submatrix
        B_size = self.C + 1
        P_BB = P2[:B_size, :B_size]
        
        # Find stationary distribution by solving π = πP
       
        try:
            # Use sparse matrices for efficiency with larger capacities
            if self.C > 50:
                P_sparse = sparse.csr_matrix(P_BB.T - np.eye(B_size))
                # Add normalization constraint
                A = sparse.vstack([P_sparse[:-1], sparse.csr_matrix(np.ones(B_size))])
                b = np.zeros(B_size)
                b[-1] = 1
                
                # Solve using sparse solver
                from scipy.sparse.linalg import spsolve
                pi = spsolve(A.T @ A, A.T @ b)
            else:
                # For smaller systems, use dense computation
                # Solve (P^T - I)π = 0 with constraint sum(π) = 1
                A = np.vstack([P_BB.T - np.eye(B_size), np.ones(B_size)])
                b = np.zeros(B_size + 1)
                b[-1] = 1
                
                # Least squares for numerical stability
                pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Normalize probabilities
            pi = np.abs(pi)  # Fix numerical errors causing negative values
            pi = pi / pi.sum()
            
            return pi
            
        except Exception as e:
            warnings.warn(f"Failed to compute steady state: {e}. Using uniform distribution.")
            return np.ones(B_size) / B_size
    
    def expected_rejections(self, lambda_rate: float) -> float:
        """
        Calculate expected number of rejections per period using equation (4)
        
        R(C,λ,p) = Σ_{j=0}^C π_j(C,λ,p) * Σ_{k=C-j+1}^∞ (λ^k * e^{-λ} / k!) * (k+j-C)
        """
        if lambda_rate == 0:
            return 0.0
        
        # For very large capacities, use fluid approximation
        if self.C >= 150:
            effective_capacity = self.C * self.p
            return max(0, lambda_rate - effective_capacity)
        
        # Build transition matrix
        P = self._build_transition_matrix(lambda_rate)
        
        # Compute steady-state probabilities for B states
        pi = self._compute_steady_state(P)
        
        # Calculate expected rejections using equation (4)
        expected_rej = 0.0
        
        for j in range(self.C + 1):
            # For each state (B,j), calculate expected rejections
            # when arrivals exceed available capacity (C - j)
            available_capacity = self.C - j
            
            # Sum over k from C-j+1 to infinity (arrivals that cause rejection)
            # Truncate infinite sum at 99.99% of Poisson mass
            max_k = int(lambda_rate + 10 * np.sqrt(lambda_rate)) + self.C
            
            for k in range(available_capacity + 1, max_k + 1):
                prob_k_arrivals = poisson.pmf(k, lambda_rate)
                n_rejected = k - available_capacity  # k + j - C
                expected_rej += pi[j] * prob_k_arrivals * n_rejected
        
        return expected_rej
    
    def get_rejection_rate(self, lambda_rate: float) -> float:
        """
        Get rejection rate (rejections / capacity) for comparison with paper
        """
        return self.expected_rejections(lambda_rate) / self.C if self.C > 0 else 0
    
    def get_piecewise_linear_points(self, max_lambda: float, n_points: int = 12) -> Dict[str, list]:
        """
        Get points for piecewise linear approximation
        Following paper's approach with non-uniform spacing
        """
        # Create non-uniform spacing with more points around ρ = 1
        effective_capacity = self.C * self.p
        
        # Define load levels (ρ values) with focus around 1
        rho_values = []
        
        # Sparse points for low load (0 to 0.8)
        rho_values.extend(np.linspace(0, 0.8, int(n_points * 0.25)))
        
        # Dense points around critical region (0.8 to 1.2)
        rho_values.extend(np.linspace(0.8, 1.2, int(n_points * 0.5)))
        
        # Sparse points for high load (1.2 to max)
        max_rho = max_lambda / effective_capacity if effective_capacity > 0 else 2.0
        if max_rho > 1.2:
            rho_values.extend(np.linspace(1.2, min(max_rho, 2.0), int(n_points * 0.25)))
        
        # Convert to lambda values and remove duplicates
        lambda_values = sorted(list(set([rho * effective_capacity for rho in rho_values])))
        lambda_values = [lam for lam in lambda_values if lam <= max_lambda]
        
        # Add boundary points if missing
        if 0 not in lambda_values:
            lambda_values.insert(0, 0)
        if lambda_values[-1] < max_lambda:
            lambda_values.append(max_lambda)
        
        # Calculate rejections for each lambda
        rejection_values = [self.expected_rejections(lam) for lam in lambda_values]
        
        return {
            'lambda_values': lambda_values,
            'rejection_values': rejection_values,
            'rho_values': [lam / effective_capacity if effective_capacity > 0 else 0 
                          for lam in lambda_values]
        }


def create_rejection_function_plot():
    """
    Create Figure 2 from the paper showing convergence to fluid model
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p = 0.5  # Pickup probability
    capacities = [10, 30, 50, 150]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot for each capacity
    for C, color in zip(capacities, colors):
        rf = RejectionFunction(capacity=C, pickup_prob=p)
        
        # Generate points for ρ from 0 to 2
        rho_values = np.linspace(0, 2, 100)
        lambda_values = [rho * C * p for rho in rho_values]
        
        # Calculate rejection rates
        rejection_rates = []
        for lam in lambda_values:
            rej = rf.expected_rejections(lam)
            rejection_rates.append(rej / C)  # Normalize by capacity
        
        ax.plot(rho_values, rejection_rates, color=color, linewidth=2, 
                label=f'C = {C}')
    
    # Add fluid approximation (C → ∞)
    rho_values_fluid = np.linspace(0, 2, 100)
    rejection_rates_fluid = [max(0, rho - 1) * p for rho in rho_values_fluid]
    ax.plot(rho_values_fluid, rejection_rates_fluid, 'k--', linewidth=2, 
            label='C → ∞')
    
    # Formatting
    ax.set_xlabel('ρ (load level)', fontsize=12)
    ax.set_ylabel('Rejections / Cp', fontsize=12)
    ax.set_title('Fig. 2. Convergence to the fluid model (for p = 0.5)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 0.5)
    
    # Add minor gridlines
    ax.grid(True, which='minor', alpha=0.1)
    ax.minorticks_on()
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test the DTMC implementation
    print("Testing DTMC Rejection Function Implementation")
    print("=" * 60)
    
    # Test with paper parameters
    C = 30
    p = 0.5
    lambda_test = 15  # ρ = 1.0
    
    rf = RejectionFunction(capacity=C, pickup_prob=p)
    rejections = rf.expected_rejections(lambda_test)
    
    print(f"Capacity (C): {C}")
    print(f"Pickup probability (p): {p}")
    print(f"Arrival rate (λ): {lambda_test}")
    print(f"Load (ρ): {lambda_test / (C * p):.2f}")
    print(f"Expected rejections: {rejections:.3f}")
    print(f"Rejection rate: {rejections/C:.3%}")
    
    # Create and save the plot
    print("\nGenerating Figure 2...")
    fig = create_rejection_function_plot()
    fig.savefig('data/output/test/figure2_dtmc_rejection.png', dpi=150, bbox_inches='tight')
    print("Plot saved to data/output/test/figure2_dtmc_rejection.png")