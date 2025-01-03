import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

def load_systems_config(config_file):
    """
    Loads the systems configuration from a JSON file.
    
    Args:
        config_file (str): Path to the JSON configuration file.
    
    Returns:
        dict: Dictionary containing system configurations.
    """
    with open(config_file, "r") as f:
        systems = json.load(f)
    return systems

def find_connected_components_loads(adjacency_matrix, generators, loads):
    """
    Finds connected components among loads.
    
    Args:
        adjacency_matrix (list of lists): Full adjacency matrix.
        generators (list): List of generator node indices.
        loads (list): List of load node indices.
    
    Returns:
        list of sets: Each set contains node indices of a connected load component.
    """
    # Create a sub-adjacency matrix for loads
    load_indices = {node: idx for idx, node in enumerate(loads)}
    num_loads = len(loads)
    load_A = [[0]*num_loads for _ in range(num_loads)]
    
    for i, load_i in enumerate(loads):
        for j, load_j in enumerate(loads):
            load_A[i][j] = adjacency_matrix[load_i][load_j]
    
    # Use NetworkX to find connected components in the load subgraph
    load_graph = nx.Graph()
    load_graph.add_nodes_from(range(num_loads))
    for i in range(num_loads):
        for j in range(i+1, num_loads):
            if load_A[i][j] == 1:
                load_graph.add_edge(i, j)
    
    connected_components = list(nx.connected_components(load_graph))
    
    # Map back to original node indices
    load_components = []
    for component in connected_components:
        load_component = set()
        for local_idx in component:
            load_node = loads[local_idx]
            load_component.add(load_node)
        load_components.append(load_component)
    
    return load_components

def create_bipartite_graph(adjacency_matrix, generators, loads, load_components):
    """
    Creates a bipartite-like graph connecting generators to load components.
    
    Args:
        adjacency_matrix (list of lists): Full adjacency matrix.
        generators (list): List of generator node indices.
        loads (list): List of load node indices.
        load_components (list of sets): Connected load components.
    
    Returns:
        networkx.Graph: Bipartite-like graph with generators and load components as nodes.
        list: Total demand for each load component.
    """
    B = nx.Graph()
    
    # Add generator nodes
    for gen in generators:
        B.add_node(f"Gen {gen}", bipartite=0, type='generator')
    
    # Add load component nodes
    for idx, comp in enumerate(load_components):
        total_demand = sum([demand_map[comp_node] for comp_node in comp])
        B.add_node(f"Comp {idx+1}", bipartite=1, type='load_component', demand=total_demand)
    
    # Connect generators to load components based on adjacency_matrix
    for gen in generators:
        for idx, comp in enumerate(load_components):
            # If generator is connected to any load in the component, connect to the component
            connected = any(adjacency_matrix[gen][load_node] == 1 for load_node in comp)
            if connected:
                B.add_edge(f"Gen {gen}", f"Comp {idx+1}")
    
    return B

def get_node_colors_and_labels(B):
    """
    Determines node colors based on type and their generation/demand values.
    
    Args:
        B (networkx.Graph): The bipartite-like graph with node attributes.
    
    Returns:
        list: List of colors for each node.
        list: List of labels for each node.
    """
    colors = []
    labels = []
    
    # Separate generators and load components
    generators = [node for node, attr in B.nodes(data=True) if attr['type'] == 'generator']
    load_components = [node for node, attr in B.nodes(data=True) if attr['type'] == 'load_component']
    
    # Normalize generation and demand for color intensity
    gen_values = [generation_map[int(node.split()[1])] for node in generators]
    load_demands = [B.nodes[node]['demand'] for node in load_components]
    
    # Handle edge cases where all generation or demands are equal
    if max(gen_values) != min(gen_values):
        gen_norm = [(val - min(gen_values)) / (max(gen_values) - min(gen_values)) for val in gen_values]
    else:
        gen_norm = [0.5 for _ in gen_values]
    
    if max(load_demands) != min(load_demands):
        load_norm = [(val - min(load_demands)) / (max(load_demands) - min(load_demands)) for val in load_demands]
    else:
        load_norm = [0.5 for _ in load_demands]
    
    # Define colormaps
    green_map = cm.get_cmap('Greens')
    red_map = cm.get_cmap('Reds')
    
    # Assign colors and labels
    for node in B.nodes():
        if B.nodes[node]['type'] == 'generator':
            gen_idx = generators.index(node)
            norm_val = gen_norm[gen_idx]
            color = green_map(norm_val)
            colors.append(color)
            label = f"{node}\nGen: {generation_map[int(node.split()[1])]}"
            labels.append(label)
        elif B.nodes[node]['type'] == 'load_component':
            load_idx = load_components.index(node)
            norm_val = load_norm[load_idx]
            color = red_map(norm_val)
            colors.append(color)
            label = f"{node}\nDemand: {B.nodes[node]['demand']}"
            labels.append(label)
        else:
            colors.append('grey')
            labels.append(node)
    
    return colors, labels

def plot_system_graph(ax, B, colors, labels, system_name):
    """
    Plots the bipartite-like graph on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        B (networkx.Graph): The bipartite-like graph.
        colors (list): List of node colors.
        labels (list): List of node labels.
        system_name (str): Name of the system for the title.
    """
    pos = nx.spring_layout(B, seed=42)  # Fixed seed for reproducibility
    
    # Draw nodes
    nx.draw_networkx_nodes(B, pos, node_color=colors, node_size=800, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(B, pos, ax=ax, width=2)
    
    # Draw labels
    nx.draw_networkx_labels(B, pos, labels=dict(zip(B.nodes(), labels)), font_size=10, font_color='black', ax=ax)
    
    # Set title
    ax.set_title(system_name, fontsize=16)
    
    # Remove axis
    ax.set_axis_off()
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=green_map(0.6), edgecolor='k', label='Generator'),
        Patch(facecolor=red_map(0.6), edgecolor='k', label='Load Component')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def main():
    # Load systems configuration
    config_file = "systems_config.json"
    if not os.path.isfile(config_file):
        print(f"Configuration file '{config_file}' not found.")
        return
    systems = load_systems_config(config_file)
    
    # Define IEEE systems to plot
    ieee_systems = ['IEEE14', 'IEEE30', 'IEEE57', 'IEEE118']
    
    # Prepare data for node coloring
    global generation_map
    global demand_map
    generation_map = {}
    demand_map = {}
    
    # Initialize for all systems to find maximum generation and demand for normalization
    for system_name in ieee_systems:
        if system_name not in systems:
            continue
        system = systems[system_name]
        generators = system["generators"]
        loads = system["loads"]
        generation = system["generation"]
        demand = system["demand"]
        for gen, gen_cap in zip(generators, generation):
            generation_map[gen] = gen_cap
        for load, dem in zip(loads, demand):
            demand_map[load] = dem
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(25, 20))
    axs = axs.flatten()
    
    for idx, system_name in enumerate(ieee_systems):
        ax = axs[idx]
        if system_name not in systems:
            print(f"Warning: System '{system_name}' not found in configuration.")
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', fontsize=20)
            ax.set_title(system_name)
            ax.set_axis_off()
            continue
        
        system = systems[system_name]
        generators = system["generators"]
        generation = system["generation"]
        loads = system["loads"]
        demand = system["demand"]
        adjacency_matrix = system["adjacency_matrix"]
        
        # Identify load components
        load_components = find_connected_components_loads(adjacency_matrix, generators, loads)
        
        # Create bipartite-like graph
        B = create_bipartite_graph(adjacency_matrix, generators, loads, load_components)
        
        # Get node colors and labels
        colors, labels = get_node_colors_and_labels(B)
        
        # Define colormaps (should be consistent with main)
        global green_map, red_map
        green_map = cm.get_cmap('Greens')
        red_map = cm.get_cmap('Reds')
        
        # Plot the graph
        plot_system_graph(ax, B, colors, labels, system_name)
    
    plt.suptitle('IEEE System Graphs Before Perturbations', fontsize=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.show()

if __name__ == "__main__":
    main()
