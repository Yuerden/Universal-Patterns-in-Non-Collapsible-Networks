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

def create_graph(adjacency_matrix, generators, loads):
    """
    Creates a NetworkX graph from the adjacency matrix.
    
    Args:
        adjacency_matrix (list of lists): Adjacency matrix representing the graph.
        generators (list): List of generator node indices.
        loads (list): List of load node indices.
    
    Returns:
        networkx.Graph: Constructed graph.
    """
    G = nx.DiGraph()  # Assuming directed graph
    
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)
    
    # Assign node attributes
    for node in G.nodes():
        if node in generators:
            G.nodes[node]['type'] = 'generator'
        elif node in loads:
            G.nodes[node]['type'] = 'load'
        else:
            G.nodes[node]['type'] = 'unknown'
    
    return G

def plot_system_graph(ax, G, colors, system_name, generators, generation, loads, demand):
    """
    Plots the graph on the given axes, labeling nodes by their generation or demand values.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        G (networkx.Graph): The graph to plot.
        colors (list): List of node colors.
        system_name (str): Name of the system for the title.
        generators (list): List of generator nodes.
        generation (list): List of generation capacities.
        loads (list): List of load nodes.
        demand (list): List of load demands.
    """
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility

    # Create a label dictionary where:
    # - Generator nodes are labeled by their generation capacity
    # - Load nodes are labeled by their demand
    labels = {}
    for node in G.nodes():
        if G.nodes[node]['type'] == 'generator':
            # Find the corresponding generation value
            gen_idx = generators.index(node)
            gen_value = generation[gen_idx]
            labels[node] = str(gen_value)
        elif G.nodes[node]['type'] == 'load':
            # Find the corresponding demand value
            load_idx = loads.index(node)
            load_value = demand[load_idx]
            labels[node] = str(load_value)
        else:
            labels[node] = ''  # unknown type, leave blank or assign as needed

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='black', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', ax=ax)
    
    ax.set_title(system_name)
    ax.set_axis_off()

def main():
    # Load systems configuration
    config_file = "systems_config.json"
    if not os.path.isfile(config_file):
        print(f"Configuration file '{config_file}' not found.")
        return
    systems = load_systems_config(config_file)
    
    # Define IEEE systems to plot
    ieee_systems = ['IEEE14', 'IEEE30', 'IEEE57', 'IEEE118']
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
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
        
        G = create_graph(adjacency_matrix, generators, loads)
        
        # Normalize generation and demand for color intensity
        max_gen = max(generation) if generators else 1
        min_gen = min(generation) if generators else 0
        max_dem = max(demand) if loads else 1
        min_dem = min(demand) if loads else 0
        
        colors = []
        for node in G.nodes():
            if node in generators:
                # Normalize generation
                norm = (generation[generators.index(node)] - min_gen) / (max_gen - min_gen) if max_gen != min_gen else 0.5
                color = cm.get_cmap('Greens')(norm)
                colors.append(color)
            elif node in loads:
                # Normalize demand
                norm = (demand[loads.index(node)] - min_dem) / (max_dem - min_dem) if max_dem != min_dem else 0.5
                color = cm.get_cmap('Reds')(norm)
                colors.append(color)
            else:
                colors.append('grey')
        
        plot_system_graph(ax, G, colors, system_name, generators, generation, loads, demand)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cm.get_cmap('Greens')(0.6), edgecolor='k', label='Generator'),
            Patch(facecolor=cm.get_cmap('Reds')(0.6), edgecolor='k', label='Load')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('IEEE System Graphs Before Perturbations', fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
