"""
Visualization utilities for the SP location problem
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd


def plot_network(instance: Dict, figsize: Tuple[int, int] = (12, 10)):
    """
    Plot the network structure showing demand points and candidate locations
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Extract data
    demand_points = instance["demand_points"]
    sp_locations = instance["sp_locations"]
    demand_rates = instance.get("demand_rates", [5.0] * len(demand_points))
    
    # Plot demand points
    dx = [p[0] for p in demand_points]
    dy = [p[1] for p in demand_points]
    
    # Size proportional to demand
    sizes = [r * 10 for r in demand_rates]
    
    scatter1 = ax.scatter(dx, dy, c=demand_rates, cmap='YlOrRd', 
                         s=sizes, alpha=0.6, edgecolors='black', 
                         linewidth=0.5, label='Demand points')
    
    # Plot SP candidate locations
    sx = [p[0] for p in sp_locations]
    sy = [p[1] for p in sp_locations]
    
    ax.scatter(sx, sy, c='blue', marker='s', s=100, 
              alpha=0.7, edgecolors='darkblue', 
              linewidth=2, label='SP candidates')
    
    # Add service radius example
    if len(sp_locations) > 0:
        # Show radius for first SP
        circle = plt.Circle(sp_locations[0], instance.get("service_radius", 600), 
                          fill=False, edgecolor='blue', linestyle='--', 
                          linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    # Formatting
    ax.set_xlabel('X coordinate (m)', fontsize=12)
    ax.set_ylabel('Y coordinate (m)', fontsize=12)
    ax.set_title('Service Point Network Structure', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Colorbar for demand
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('Demand rate', fontsize=10)
    
    # Legend
    ax.legend(loc='upper right')
    
    # Add statistics
    stats_text = (
        f"Demand points: {len(demand_points)}\n"
        f"SP locations: {len(sp_locations)}\n"
        f"Total demand: {sum(demand_rates):.1f}\n"
        f"Service radius: {instance.get('service_radius', 600)}m"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_solution(instance: Dict, solution: Dict, 
                 figsize: Tuple[int, int] = (14, 10),
                 save_path: Optional[str] = None):
    """
    Plot the solution showing opened SPs and allocations
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract data
    demand_points = instance["demand_points"]
    sp_locations = instance["sp_locations"]
    demand_rates = instance.get("demand_rates", [5.0] * len(demand_points))
    
    # Opened SPs
    opened_sps = solution.get("service_points", [])
    
    # --- Left plot: Network with solution ---
    # Plot demand points
    dx = [p[0] for p in demand_points]
    dy = [p[1] for p in demand_points]
    
    ax1.scatter(dx, dy, c='lightgray', s=30, alpha=0.5, label='Demand points')
    
    # Plot all SP candidates
    sx = [p[0] for p in sp_locations]
    sy = [p[1] for p in sp_locations]
    ax1.scatter(sx, sy, c='lightblue', marker='s', s=50, 
               alpha=0.3, label='SP candidates (not selected)')
    
    # Plot opened SPs with size proportional to capacity
    for sp in opened_sps:
        loc = sp["location"]
        cap = sp["capacity"]
        util = sp.get("utilization", 0.0)
        
        # Color based on utilization
        color = plt.cm.RdYlGn_r(util)  # Red for high utilization
        
        ax1.scatter(loc[0], loc[1], c=[color], marker='s', 
                   s=cap*3, edgecolors='black', linewidth=2)
        
        # Add capacity label
        ax1.annotate(f'C={cap}', (loc[0], loc[1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8)
    
    # Plot allocations
    for alloc in solution.get("allocations", []):
        d_idx = alloc["demand_point"]
        sp_idx = alloc["service_point"]
        flow = alloc["flow"]
        
        d_loc = demand_points[d_idx]
        sp_loc = sp_locations[sp_idx]
        
        # Line width proportional to flow
        ax1.plot([d_loc[0], sp_loc[0]], [d_loc[1], sp_loc[1]], 
                'k-', alpha=0.3, linewidth=flow/5)
    
    ax1.set_xlabel('X coordinate (m)')
    ax1.set_ylabel('Y coordinate (m)')
    ax1.set_title('Solution: Opened SPs and Allocations')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # --- Right plot: Statistics ---
    if solution.get("summary"):
        summary = solution["summary"]
        
        # Utilization distribution
        utils = [sp["utilization"] for sp in opened_sps]
        
        ax2.hist(utils, bins=10, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(utils), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(utils):.2f}')
        ax2.set_xlabel('Utilization')
        ax2.set_ylabel('Number of SPs')
        ax2.set_title('SP Utilization Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = (
            f"Opened SPs: {summary['num_service_points']}\n"
            f"Total capacity: {summary['total_capacity']}\n"
            f"Total demand: {summary['total_demand']:.1f}\n"
            f"Avg utilization: {summary['avg_utilization']:.2%}\n"
            f"Setup cost: ${summary['total_setup_cost']:,.0f}\n"
            f"Rejection cost: ${summary['total_rejection_cost']:,.0f}\n"
            f"Total cost: ${solution['objective_value']:,.0f}"
        )
        
        ax2.text(0.5, 0.3, stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f"Service Point Location Solution (Gap: {solution.get('mip_gap', 0):.1%})", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_cost_breakdown(solution: Dict, figsize: Tuple[int, int] = (10, 6)):
    """
    Plot cost breakdown of the solution
    """

    if not solution.get("summary"):
        print("No summary data available")
        return
    
    summary = solution["summary"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart of costs
    costs = [summary["total_setup_cost"], summary["total_rejection_cost"]]
    labels = ['Setup Cost', 'Rejection Cost']
    colors = ['#3498db', '#e74c3c']
    
    ax1.pie(costs, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12})
    ax1.set_title('Cost Breakdown', fontsize=14, fontweight='bold')
    
    # Bar chart by SP
    sp_costs = []
    sp_labels = []
    
    for i, sp in enumerate(solution["service_points"]):
        sp_costs.append(sp["setup_cost"])
        sp_labels.append(f"SP{i+1}\n(C={sp['capacity']})")
    
    ax2.bar(range(len(sp_costs)), sp_costs, color='#3498db', edgecolor='black')
    ax2.set_xticks(range(len(sp_costs)))
    ax2.set_xticklabels(sp_labels, rotation=45)
    ax2.set_ylabel('Setup Cost ($)')
    ax2.set_title('Setup Cost by Service Point', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_rejection_function_validation(output_path):
    """
    Create Figure 2 from the paper showing convergence to fluid model
    Shows rejection rate as function of load (rho) for different capacities
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from models.rejection_function import RejectionFunction
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Parameters
    p = 0.5  # pickup probability
    capacities = [10, 30, 50, 150]  # Different capacity values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for each capacity
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot for each capacity
    for i, C in enumerate(capacities):
        # Create range of rho values (load level)
        rho_values = np.linspace(0, 2, 500)
        rejection_rates = []
        
        for rho in rho_values:
            # Lambda = rho * C * p
            lam = rho * C * p
            
            # Calculate rejection rate using the rejection function
            rf = RejectionFunction(C, p)
            rejections = rf.expected_rejections(lam)
            
            # Rejection rate = rejections / (C * p)
            rejection_rate = rejections / (C * p) if C * p > 0 else 0
            rejection_rates.append(rejection_rate)
        
        # Plot
        ax.plot(rho_values, rejection_rates, 
                label=f'C = {C}', 
                color=colors[i], 
                linewidth=2.5)
    
    # Add fluid model (C → ∞)
    rho_fluid = np.linspace(1, 2, 200)
    rejection_fluid = rho_fluid - 1  # For rho > 1, rejection rate = rho - 1
    ax.plot(rho_fluid, rejection_fluid, 
            'k--', 
            label='C → ∞', 
            linewidth=2.5)
    
    # Formatting
    ax.set_xlabel('ρ (load level)', fontsize=14)
    ax.set_ylabel('Rejections / Cp', fontsize=14)
    ax.set_title('Fig. 2. Convergence to the fluid model (for p = 0.5)', fontsize=16)
    
    # Set axis limits
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 0.5)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(fontsize=12, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Rejection function plot saved to {output_path}")


def plot_model_comparison(comparison_results: Dict, save_path: Optional[str] = None):
    """
    Create Figure 4 from Raviv (2023): Comparison between models
    Shows extra cost % for deterministic and scenarios-based models vs stochastic
    
    Parameters:
    -----------
    comparison_results : dict
        Results in format: {
            'scenarios': [(radius1, rej_cost1, extra_cost1), ...],
            'deterministic_extra': [extra_cost1, ...],
            'scenarios_extra': [extra_cost1, ...]
        }
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    scenarios = comparison_results['scenarios']
    deterministic_extra = comparison_results['deterministic_extra']
    scenarios_extra = comparison_results.get('scenarios_extra', [0] * len(scenarios))
    
    # Bar positions
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, deterministic_extra, width, 
                    label='Deterministic', color='#1f77b4', edgecolor='black')
    
    if scenarios_extra and any(v > 0 for v in scenarios_extra):
        bars2 = ax.bar(x + width/2, scenarios_extra, width, 
                        label='Scenarios based', color='#ff7f0e', edgecolor='black')
    
    # Formatting
    ax.set_ylabel('% Extra cost of the alternative models', fontsize=12)
    ax.set_xlabel('Rejection Cost / Radius', fontsize=12)
    ax.set_ylim(0, max(max(deterministic_extra, default=0), 
                       max(scenarios_extra, default=0)) * 1.2)
    
    # X-axis labels
    labels = []
    for radius, rej_cost, _ in scenarios:
        labels.append(f'{rej_cost}\n{radius}m')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.title('Fig. 4. Comparison with the alternative models\n' + 
              'small instances with m = 25, n = 121', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    return fig


def create_summary_report(instance: Dict, solution: Dict, 
                         output_path: str = "solution_report.png"):
    """
    Create a comprehensive summary report
    """
   
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main solution plot
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Plot solution on main axis
    demand_points = instance["demand_points"]
    sp_locations = instance["sp_locations"]
    
    # Similar plotting code as in plot_solution...
    # (abbreviated for space)
    
    ax_main.set_title('Service Point Network Solution', fontsize=16, fontweight='bold')
    
    # Statistics panels
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_costs = fig.add_subplot(gs[1, 2])
    ax_util = fig.add_subplot(gs[2, :])
    
    # Remove axes for text panels
    ax_stats.axis('off')
    ax_costs.axis('off')
    
    # Add statistics text
    if solution.get("summary"):
        summary = solution["summary"]
        
        stats_text = (
            "SOLUTION STATISTICS\n"
            "─" * 20 + "\n"
            f"Status: {'Optimal' if solution['status'] == 2 else 'Feasible'}\n"
            f"Runtime: {solution['runtime']:.1f}s\n"
            f"MIP Gap: {solution.get('mip_gap', 0):.2%}\n"
            f"\nNETWORK\n"
            f"Service Points: {summary['num_service_points']}\n"
            f"Total Capacity: {summary['total_capacity']}\n"
            f"Demand Points: {instance['n_demand_points']}\n"
            f"Total Demand: {summary['total_demand']:.1f}\n"
            f"Coverage: {summary['avg_utilization']:.1%}"
        )
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     verticalalignment='top', fontsize=10,
                     fontfamily='monospace')
        
        costs_text = (
            "COST ANALYSIS\n"
            "─" * 20 + "\n"
            f"Setup Cost: ${summary['total_setup_cost']:,.0f}\n"
            f"Rejection Cost: ${summary['total_rejection_cost']:,.0f}\n"
            f"Total Cost: ${solution['objective_value']:,.0f}\n"
            f"\nPER SP:\n"
            f"Avg Setup: ${summary['total_setup_cost']/summary['num_service_points']:,.0f}\n"
            f"Avg Capacity: {summary['total_capacity']/summary['num_service_points']:.0f}"
        )
        
        ax_costs.text(0.1, 0.9, costs_text, transform=ax_costs.transAxes,
                     verticalalignment='top', fontsize=10,
                     fontfamily='monospace')
    
    # Utilization chart
    opened_sps = solution.get("service_points", [])
    positions = range(len(opened_sps))
    utils = [sp["utilization"] for sp in opened_sps]
    capacities = [sp["capacity"] for sp in opened_sps]
    
    bars = ax_util.bar(positions, utils, color=['green' if u < 0.8 else 'orange' if u < 0.95 else 'red' for u in utils])
    ax_util.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax_util.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
    ax_util.set_ylabel('Utilization')
    ax_util.set_xlabel('Service Point')
    ax_util.set_title('Service Point Utilization', fontsize=12)
    ax_util.set_ylim(0, 1.2)
    
    # Add capacity labels
    for i, (bar, cap) in enumerate(zip(bars, capacities)):
        height = bar.get_height()
        ax_util.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'C={cap}', ha='center', va='bottom', fontsize=8)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary report saved to {output_path}")
    
    return fig