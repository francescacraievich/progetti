import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from network_parsers import get_node_labels

def visualize_network_with_solution(G, capacities, result, optimizer, pos=None):
    """Visualization showing traffic flows and utilization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    if pos is None:
        # kamada kawai layout for better visualization
        pos = nx.kamada_kawai_layout(G, scale=2)
    
    # Ensure consistent node ordering
    node_list = list(G.nodes())
    
    # Get node labels
    labels = get_node_labels(G)
    
    # Calculate IGP-only utilizations for comparison
    igp_util, _, igp_arc_utils = optimizer.compute_igp_only_utilization()
    
    # Left plot: IGP-only utilization
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1500, ax=ax1)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax1)
    
    # Color edges based on IGP utilization
    edges_to_draw = []
    processed_pairs = set()
    
    for u, v in G.edges():
        # Create a canonical representation of the edge (sorted tuple)
        edge_key = tuple(sorted([u, v]))
        if edge_key in processed_pairs:
            continue
            
        # Get max utilization for this edge (both directions)
        util1 = igp_arc_utils.get((u, v), 0)
        util2 = igp_arc_utils.get((v, u), 0)
        max_util = max(util1, util2)
        
        # Color coding: green < 50%, yellow 50-80%, orange 80-100%, red > 100%
        if max_util < 50:
            color = 'green'
        elif max_util < 80:
            color = 'yellow'
        elif max_util < 100:
            color = 'orange'
        else:
            color = 'red'
        
        # Width based on utilization
        width = 1 + max_util / 20
        edges_to_draw.append(((u, v), color, width))
        processed_pairs.add(edge_key)
    
    # Draw all edges - sort by utilization so higher utilization is drawn last
    edges_to_draw.sort(key=lambda x: x[2])
    for (u, v), color, width in edges_to_draw:
        nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=color, width=width, 
                               alpha=0.9, ax=ax1, arrows=False,
                               connectionstyle="arc3,rad=0.1")
    
    # Add utilization labels (only show max utilization for each edge)
    edge_labels_igp = {}
    processed_edges = set()
    
    for u, v in G.edges():
        # Skip if we already processed this edge in reverse
        if (v, u) in processed_edges:
            continue
            
        util1 = igp_arc_utils.get((u, v), 0)
        util2 = igp_arc_utils.get((v, u), 0)
        max_util = max(util1, util2)
        if max_util > 0:
            edge_labels_igp[(u, v)] = f"{max_util:.0f}%"
        processed_edges.add((u, v))
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels_igp, font_size=7, ax=ax1)
    
    ax1.set_title(f"IGP-Only Routing (Max Util: {igp_util:.1f}%)", fontsize=14)
    ax1.axis('off')
    
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
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax2)
        
        # Color edges based on combined utilization
        edges_to_draw = []
        processed_pairs = set()
        
        # First, identify all bidirectional edges and their utilizations
        edge_utilizations = {}
        for u, v in G.edges():
            util_forward = combined_arc_utils.get((u, v), 0)
            util_backward = combined_arc_utils.get((v, u), 0)
            edge_utilizations[(u, v)] = (util_forward, util_backward)
        
        for u, v in G.edges():
            # Create a canonical representation of the edge (sorted tuple)
            edge_key = tuple(sorted([u, v]))
            if edge_key in processed_pairs:
                continue
                
            # Get utilizations for both directions
            util_forward, util_backward = edge_utilizations.get((u, v), (0, 0))
            
            # Use the maximum utilization for coloring
            max_util = max(util_forward, util_backward)
            
            if max_util < 50:
                color = 'green'
            elif max_util < 80:
                color = 'yellow'
            elif max_util < 100:
                color = 'orange'
            else:
                color = 'red'
            
            width = 1 + max_util / 20
            edges_to_draw.append(((u, v), color, width, max_util))
            processed_pairs.add(edge_key)
        
        # Draw all edges - sort by utilization so higher utilization is drawn last
        edges_to_draw.sort(key=lambda x: x[3])
        for (u, v), color, width, _ in edges_to_draw:
            nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=color, width=width, 
                                   alpha=0.95, ax=ax2, arrows=False)
        
        # Add utilization labels (only show max utilization for each edge)
        edge_labels_combined = {}
        processed_edges = set()
        
        for u, v in G.edges():
            # Skip if we already processed this edge in reverse
            if (v, u) in processed_edges:
                continue
                
            util1 = combined_arc_utils.get((u, v), 0)
            util2 = combined_arc_utils.get((v, u), 0)
            max_util = max(util1, util2)
            if max_util > 0:
                edge_labels_combined[(u, v)] = f"{max_util:.0f}%"
            processed_edges.add((u, v))
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels_combined, font_size=7, ax=ax2)
        
        # Add title based on survivability
        if 'survivability' in result and result['survivability']:
            title = f"Combined Solution with Survivability (Max Util: {result['u_max_percent']:.1f}%)"
        else:
            title = f"Combined Solution (Max Util: {result['u_max_percent']:.1f}%)"
        ax2.set_title(title, fontsize=14)
    else:
        ax2.text(0.5, 0.5, "No Solution Found", ha='center', va='center', 
                fontsize=20, transform=ax2.transAxes)
    
    ax2.axis('off')
    
    # Add legend in the middle
    legend_elements = [
        Patch(facecolor='green', label='< 50%'),
        Patch(facecolor='yellow', label='50-80%'),
        Patch(facecolor='orange', label='80-100%'),
        Patch(facecolor='red', label='> 100%')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, frameon=True, 
               bbox_to_anchor=(0.5, 0.02))
    
    # Add overall statistics
    if result['status'] == 'optimal':
        fig.text(0.5, 0.05, f"Total Demand: {sum(optimizer.demands.values()):.0f} Mbps | "
                           f"IGP Traffic: {result['total_igp']:.0f} Mbps ({result['igp_percent']:.1f}%) | "
                           f"MPLS Traffic: {result['total_mpls']:.0f} Mbps ({result['mpls_percent']:.1f}%) | "
                           f"LSPs: {result['lsp_count']}",
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def visualize_failure_scenario(G, capacities, optimizer, failed_link, result_with_survivability, pos=None):
    """Visualize network utilization under a specific link failure with survivability solution"""
    if pos is None:
        pos = nx.kamada_kawai_layout(G, scale=2)
    
    plt.figure(figsize=(12, 10))
    
    # Get node labels
    labels = get_node_labels(G)
    
    # Compute failure routing matrix
    failure_routing = optimizer.compute_failure_routing_matrix(failed_link)
    
    # Calculate utilizations with failed link including MPLS from survivability solution
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
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=1500)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Draw edges with failure scenario coloring
    edge_list = []
    edge_colors = []
    edge_widths = []
    edge_styles = []
    
    for u, v in G.edges():
        if (u, v) == failed_link or (v, u) == failed_link:
            # Failed link in black/dashed
            edge_list.append((u, v))
            edge_colors.append('black')
            edge_widths.append(3)
            edge_styles.append('dashed')
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
            
            edge_list.append((u, v))
            edge_colors.append(color)
            edge_widths.append(1 + max_util / 20)
            edge_styles.append('solid')
    
    # Draw all edges at once
    for i, (u, v) in enumerate(edge_list):
        nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=edge_colors[i], 
                              width=edge_widths[i], style=edge_styles[i])
    
    
    edge_labels = {}
    processed_edges = set()
    
    for u, v in G.edges():
        # Skip if we already processed this edge in reverse
        if (v, u) in processed_edges:
            continue
            
        if (u, v) == failed_link or (v, u) == failed_link:
            edge_labels[(u, v)] = "FAILED"
        else:
            util1 = arc_utils_failure.get((u, v), 0)
            util2 = arc_utils_failure.get((v, u), 0)
            max_util = max(util1, util2)
            if max_util > 0:
                edge_labels[(u, v)] = f"{max_util:.0f}%"
        processed_edges.add((u, v))
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    # Find max utilization
    max_util_failure = max(arc_utils_failure.values()) if arc_utils_failure else 0
    
    # Get network name from node naming convention
    network_name = "Network"
    if any('N' in str(node) for node in G.nodes()):
        network_name = "Atlanta Network"
    elif any('.' in str(node) for node in G.nodes()):
        network_name = "Geant Network"
    
    # Get proper link name for display
    if labels:
        src_label = labels.get(failed_link[0], failed_link[0])
        dst_label = labels.get(failed_link[1], failed_link[1])
        link_display = f"{src_label} - {dst_label}"
    else:
        link_display = f"{failed_link[0]} - {failed_link[1]}"
    
    # Add more space at the top for title
    plt.suptitle(f"{network_name} Under Failure: Link {link_display} Failed\n"
                 f"Max Utilization: {max_util_failure:.1f}%", fontsize=16, y=0.98)
    
    # Add legend
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
    plt.subplots_adjust(top=0.9)  
    plt.show()

def visualize_lsp_flows(G, result, optimizer, pos=None):
    """Visualize the LSP flows in the network"""
    if result['status'] != 'optimal' or result['lsp_count'] == 0:
        print("No LSP flows to visualize")
        return
    
    if pos is None:
        pos = nx.kamada_kawai_layout(G, scale=2)
    
    plt.figure(figsize=(14, 10))
    
    # Get node labels
    labels = get_node_labels(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1200)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Draw base edges 
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5)
    
    # Extract LSP information from full paths
    edge_flows = {}
    
    # Use full paths if available
    if 'lsp_full_paths' in result:
        for src, dst, path, flow in result['lsp_full_paths']:
            # Add flow to each edge in the path
            for i in range(len(path) - 1):
                edge = tuple(sorted([path[i], path[i+1]]))
                edge_flows[edge] = edge_flows.get(edge, 0) + flow
    else:
        # Fallback to lsp_details
        for src, dst, flow in result['lsp_details']:
            edge = tuple(sorted([src, dst]))
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
    

    legend_text = [
        f"Total LSPs: {total_lsps}",
        f"Total MPLS Traffic: {total_bandwidth:.0f} Mbps",
        "─" * 25,
        "Top LSPs:"
    ]
    
    # Show top LSP details with proper labels
    if 'lsp_full_paths' in result:
        for i, (src, dst, path, flow) in enumerate(result['lsp_full_paths'][:8], 1):
            src_label = labels.get(src, src)
            dst_label = labels.get(dst, dst)
            legend_text.append(f"{i}. {src_label} → {dst_label}: {flow:.0f} Mbps")
    else:
        for i, (src, dst, flow) in enumerate(result['lsp_details'][:8], 1):
            src_label = labels.get(src, src)
            dst_label = labels.get(dst, dst)
            legend_text.append(f"{i}. {src_label} → {dst_label}: {flow:.0f} Mbps")
    
    if total_lsps > 8:
        legend_text.append(f"... and {total_lsps - 8} more LSPs")
    
   
    textstr = '\n'.join(legend_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top', bbox=props)
    
    # Determine network name
    network_name = "Network"
    if any('N' in str(node) for node in G.nodes()):
        network_name = "Atlanta Network"
    elif any('.' in str(node) for node in G.nodes()):
        network_name = "Geant Network"
    
    # Add title
    if 'survivability' in result and result['survivability']:
        title = f'{network_name}: LSP Flows with Survivability\n(Total {total_lsps} LSPs, {total_bandwidth:.0f} Mbps)'
    else:
        title = f'{network_name}: LSP Flows\n(Total {total_lsps} LSPs, {total_bandwidth:.0f} Mbps)'
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_igp_weights_comparison(G, capacities, demands, default_weights, optimized_weights, pos=None):
    """Visualize the effect of IGP weight optimization comparing default vs optimized weights"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    if pos is None:
        pos = nx.kamada_kawai_layout(G, scale=2)
    
    # Get node labels
    labels = get_node_labels(G)
    
    # Left plot: Default weights (all = 1)
    ax1.set_title("Default IGP Weights (All = 1)\nSimple but often inefficient", fontsize=14)
    
    # Compute routing and utilization with default weights
    from network_parsers import compute_ospf_routing
    default_routing = compute_ospf_routing(G, demands)
    from collections import defaultdict
    default_arc_loads = defaultdict(float)
    for arc in [(u,v) for u,v in G.edges()] + [(v,u) for u,v in G.edges()]:
        load = 0
        for f in demands:
            if (f, arc) in default_routing:
                load += demands[f] * default_routing[f, arc]
        default_arc_loads[arc] = load
    
    # Calculate utilizations
    default_arc_utils = {}
    for arc, load in default_arc_loads.items():
        if arc in capacities and capacities[arc] > 0:
            default_arc_utils[arc] = (load / capacities[arc]) * 100
    
    # Draw network with default weights
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=1500, ax=ax1)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax1)
    

    edges_to_draw = []
    processed_pairs = set()
    
    # Identify all bidirectional edges and their utilizations
    edge_utilizations = {}
    for u, v in G.edges():
        util_forward = default_arc_utils.get((u, v), 0)
        util_backward = default_arc_utils.get((v, u), 0)
        edge_utilizations[(u, v)] = (util_forward, util_backward)
    
    for u, v in G.edges():
        # Create a canonical representation of the edge 
        edge_key = tuple(sorted([u, v]))
        if edge_key in processed_pairs:
            continue
            
        # Get utilizations for both directions
        util_forward, util_backward = edge_utilizations.get((u, v), (0, 0))
        max_util = max(util_forward, util_backward)
        
        if max_util < 50:
            color = 'green'
        elif max_util < 80:
            color = 'yellow'
        elif max_util < 100:
            color = 'orange'
        else:
            color = 'red'
        
        width = 1 + max_util / 20
        edges_to_draw.append(((u, v), color, width, max_util))
        processed_pairs.add(edge_key)
    

    edges_to_draw.sort(key=lambda x: x[3])
    for (u, v), color, width, _ in edges_to_draw:
        nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=color, width=width, 
                               alpha=0.95, ax=ax1, arrows=False)
    
    # Show utilization percentages
    edge_utils_default = {}
    processed_edges = set()
    
    for u, v in G.edges():
        if (v, u) in processed_edges:
            continue
        util1 = default_arc_utils.get((u, v), 0)
        util2 = default_arc_utils.get((v, u), 0)
        max_util = max(util1, util2)
        if max_util > 0:
            edge_utils_default[(u, v)] = f"{max_util:.0f}%"
        processed_edges.add((u, v))
    
    nx.draw_networkx_edge_labels(G, pos, edge_utils_default, font_size=7, ax=ax1)
    
    # Add max utilization
    max_util_default = max(default_arc_utils.values()) if default_arc_utils else 0
    ax1.text(0.02, 0.98, f"Max Utilization: {max_util_default:.1f}%", 
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Right plot: Optimized weights
    ax2.set_title("Tabu Search Optimized IGP Weights\nBalances load across network", fontsize=14)
    
    # Compute routing with optimized weights
    G_weighted = G.copy()
    for u, v in G_weighted.edges():
        G_weighted[u][v]['weight'] = optimized_weights.get((u, v), 1)
    
    optimized_routing = {}
    for (s, t), demand in demands.items():
        try:
            path = nx.shortest_path(G_weighted, s, t, weight='weight')
            for i in range(len(path) - 1):
                arc = (path[i], path[i+1])
                optimized_routing[(s,t), arc] = 1.0
        except nx.NetworkXNoPath:
            pass
    
    # Calculate loads and utilizations
    optimized_arc_loads = defaultdict(float)
    for arc in [(u,v) for u,v in G.edges()] + [(v,u) for u,v in G.edges()]:
        load = 0
        for f in demands:
            if (f, arc) in optimized_routing:
                load += demands[f] * optimized_routing[f, arc]
        optimized_arc_loads[arc] = load
    
    optimized_arc_utils = {}
    for arc, load in optimized_arc_loads.items():
        if arc in capacities and capacities[arc] > 0:
            optimized_arc_utils[arc] = (load / capacities[arc]) * 100
    
    # Draw network with optimized weights
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=1500, ax=ax2)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax2)
    
    edges_to_draw = []
    processed_pairs = set()
    
    # Identify all bidirectional edges and their utilizations
    edge_utilizations = {}
    for u, v in G.edges():
        util_forward = optimized_arc_utils.get((u, v), 0)
        util_backward = optimized_arc_utils.get((v, u), 0)
        edge_utilizations[(u, v)] = (util_forward, util_backward)
    
    for u, v in G.edges():
        # Create a canonical representation of the edge 
        edge_key = tuple(sorted([u, v]))
        if edge_key in processed_pairs:
            continue
            
        # Get utilizations for both directions
        util_forward, util_backward = edge_utilizations.get((u, v), (0, 0))
        max_util = max(util_forward, util_backward)
        
        if max_util < 50:
            color = 'green'
        elif max_util < 80:
            color = 'yellow'
        elif max_util < 100:
            color = 'orange'
        else:
            color = 'red'
        
        width = 1 + max_util / 20
        edges_to_draw.append(((u, v), color, width, max_util))
        processed_pairs.add(edge_key)
    
   
    edges_to_draw.sort(key=lambda x: x[3])
    for (u, v), color, width, _ in edges_to_draw:
        nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=color, width=width, 
                               alpha=0.95, ax=ax2, arrows=False)
    

    edge_labels_optimized = {}
    processed_edges = set()
    
    for u, v in G.edges():
        if (v, u) in processed_edges:
            continue
        util1 = optimized_arc_utils.get((u, v), 0)
        util2 = optimized_arc_utils.get((v, u), 0)
        max_util = max(util1, util2)
        
    
        if max_util > 0:
            edge_labels_optimized[(u, v)] = f"{max_util:.0f}%"
        
        processed_edges.add((u, v))
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels_optimized, font_size=7, ax=ax2)
    
    # Add max utilization
    max_util_optimized = max(optimized_arc_utils.values()) if optimized_arc_utils else 0
    ax2.text(0.02, 0.98, f"Max Utilization: {max_util_optimized:.1f}%", 
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.axis('off')
    ax2.axis('off')
    

    legend_elements = [
        Patch(facecolor='green', label='< 50%'),
        Patch(facecolor='yellow', label='50-80%'),
        Patch(facecolor='orange', label='80-100%'),
        Patch(facecolor='red', label='> 100%')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True, 
               bbox_to_anchor=(0.5, 0.01))
    
    # Add overall comparison
    improvement = ((max_util_default - max_util_optimized) / max_util_default * 100) if max_util_default > 0 else 0
    fig.text(0.5, 0.08, f"IGP Weight Optimization Result: {max_util_default:.1f}% → {max_util_optimized:.1f}% (Improvement: {improvement:.1f}%)",
            ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

def visualize_simple_vs_iterative_comparison(network_name, simple_result, iterative_result, igp_util):
    """Visualize comparison between simple and iterative approach"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Utilization comparison
    approaches = ['IGP-only', 'Simple\n(Paper)', 'Iterative\n(Mine)']
    utilizations = [igp_util, simple_result['u_max_percent'], iterative_result['u_max_percent']]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    bars1 = ax1.bar(approaches, utilizations, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, util in zip(bars1, utilizations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{util:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Maximum Link Utilization (%)', fontsize=12)
    ax1.set_title(f'{network_name} Network: Maximum Utilization Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(utilizations) * 1.15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add improvement annotations
    if simple_result['u_max_percent'] > iterative_result['u_max_percent']:
        improvement = simple_result['u_max_percent'] - iterative_result['u_max_percent']
        ax1.annotate(f'↓ {improvement:.1f}%', 
                    xy=(2, iterative_result['u_max_percent']), 
                    xytext=(2.3, (simple_result['u_max_percent'] + iterative_result['u_max_percent'])/2),
                    fontsize=11, color='green', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Right plot: LSP count comparison
    lsp_counts = [0, simple_result['lsp_count'], iterative_result['lsp_count']]
    bars2 = ax2.bar(approaches, lsp_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars2, lsp_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Number of LSPs', fontsize=12)
    ax2.set_title(f'{network_name} Network: LSP Count Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(lsp_counts) * 1.15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add note about paper findings
    fig.text(0.5, 0.02, "Note: Contrary to Cherubini et al. findings, my iterative implementation achieves significant improvements", 
             ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()