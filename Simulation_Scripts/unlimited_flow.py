import json
import numpy as np
import random
import csv
import os

# Example: Node 0 and 1 are generators, Node 2-5 are loads
# generators = [0, 1]
# generation = [100, 100]  # Generator 0 produces 50, Generator 1 produces 40
# loads = [2, 3, 4, 5, 6]
# demand = [70, 10, 25, 15, 50]  # Load 2 requires 10, Load 3 requires 10, etc.

# adjacency_matrix = np.array([
#     [0, 0, 1, 0, 0, 0, 1],  # Node 0 (Generator)
#     [0, 0, 0, 1, 0, 0, 0],  # Node 1 (Generator)
#     [1, 0, 0, 1, 0, 0, 0],  # Node 2 (Load)
#     [0, 1, 1, 0, 1, 0, 0],  # Node 3 (Load)
#     [0, 0, 0, 1, 0, 1, 0],  # Node 4 (Load)
#     [0, 0, 0, 0, 1, 0, 0],   # Node 5 (Load)
#     [1, 0, 0, 0, 0, 0, 0]   # Node 6 (Load)
# ])


# generators = [0, 1, 2]
# generation = [100, 100, 100]  # Generator 0 produces 50, Generator 1 produces 40
# loads = [3, 4, 5, 6]
# demand = [80, 140, 80, 10]  # Load 2 requires 10, Load 3 requires 10, etc.

# adjacency_matrix = np.array([
#     [0, 0, 0, 1, 1, 1, 0],  # Node 0 (Generator)
#     [0, 0, 0, 0, 0, 1, 1],  # Node 1 (Generator)
#     [0, 0, 0, 1, 1, 0, 0],  # Node 2 (Generator)
#     [0, 0, 0, 0, 0, 0, 0],  # Node 3 (Load)
#     [0, 0, 0, 0, 0, 0, 0],  # Node 4 (Load)
#     [0, 0, 0, 0, 0, 0, 0],  # Node 5 (Load)
#     [0, 0, 0, 0, 0, 0, 0]   # Node 6 (Load)
# ])


import json

with open("systems_config.json", "r") as f:
    loaded_systems = json.load(f)

SYSTEM_NAME = "IEEE14" # IEEE14,30,57,118

# ============================
# Define Helper Functions
# ============================

def find_connected_components(adjacency_matrix):
    """
    Finds connected components in a graph represented by an adjacency matrix.

    Args:
        adjacency_matrix: 2D matrix where 1=link exists, 0=no link.

    Returns:
        List of connected components, where each component is a set of node indices.
    """
    num_nodes = len(adjacency_matrix)
    visited = [False] * num_nodes
    components = []

    def dfs(node, component):
        visited[node] = True
        component.add(node)
        for neighbor, is_connected in enumerate(adjacency_matrix[node]):
            if is_connected and not visited[neighbor]:
                dfs(neighbor, component)

    for node in range(num_nodes):
        if not visited[node]:
            component = set()
            dfs(node, component)
            components.append(component)

    return components

def load_demand_by_component(adjacency_matrix, generators, loads, demand):
    """
    Finds connected components among loads and calculates their total demand.

    Args:
        adjacency_matrix: Full adjacency matrix (1=link exists, 0=no link).
        generators: List of generator node indices.
        loads: List of load node indices.
        demand: List of load demands.

    Returns:
        components: A list of connected components, each as a set of load indices.
        total_demand_per_component: A list of total power demands for each connected component.
    """
    # Extract the subgraph of loads
    load_A = np.zeros((len(loads), len(loads)), dtype=int)
    for i, node in enumerate(range(len(generators), len(adjacency_matrix))):
        for j, neighbor in enumerate(range(len(generators), len(adjacency_matrix))):
            load_A[i][j] = adjacency_matrix[node][neighbor]

    # Find connected components in the load subgraph
    components = find_connected_components(load_A)

    # Map components back to global indices
    global_components = []
    for component in components:
        global_component = {loads[node] for node in component}  # Map local indices to global indices
        global_components.append(global_component)

    # Calculate the total demand for each component
    total_demand_per_component = []
    for component in global_components:
        total_demand = sum(demand[loads.index(load)] for load in component)
        total_demand_per_component.append(total_demand)

    return global_components, total_demand_per_component

def see_component_stats(adjacency_matrix, generators, loads, demand):
    components, total_demand_per_component = load_demand_by_component(adjacency_matrix, generators, loads, demand)

    # Print results
    for i, (component, total_demand) in enumerate(zip(components, total_demand_per_component)):
        print(f"Component {i + 1}: Nodes {component}, Total Demand = {total_demand}")

def create_bipartite_graph(adjacency_matrix, generators, loads, demand):
    """
    Reduces the graph into a bipartite graph of generators and load components.

    Args:
        adjacency_matrix: Full adjacency matrix (1=link exists, 0=no link).
        generators: List of generator node indices.
        loads: List of load node indices.
        demand: List of load demands.

    Returns:
        bipartite_graph: Dictionary representing the bipartite graph.
                         Keys are generators, values are lists of connected load components.
        component_demands: List of total demands for each load component.
    """
    # Step 1: Identify connected components of loads
    components, total_demand_per_component = load_demand_by_component(adjacency_matrix, generators, loads, demand)

    # Step 2: Create bipartite graph
    bipartite_graph = {gen: [] for gen in generators}
    component_demands = []

    for comp_index, component in enumerate(components):
        component_demands.append(total_demand_per_component[comp_index])
        for gen in generators:
            # Check if the generator is connected to any load in the component
            for load in component:
                if adjacency_matrix[gen][load] == 1:
                    bipartite_graph[gen].append(comp_index)
                    break  # No need to check other loads in this component

    return bipartite_graph, component_demands

def calculate_B_eff(A_matrix, load_states):
    """
    Calculate B_eff = (1^T * A * s_in) / (1^T * A * 1)

    Args:
        A_matrix: Adjacency matrix as a 2D list or NumPy array
        load_states: dict mapping load node indices to their states

    Returns:
        B_eff: float
    """
    A = np.array(A_matrix)
    num_nodes = A.shape[0]
    s_in = np.zeros(num_nodes)
    for load_node, state in load_states.items():
        s_in[load_node] = state
    numerator = np.sum(A * s_in)
    denominator = np.sum(A)
    if denominator == 0:
        return 0
    B_eff = numerator / denominator
    return B_eff

def record_data(file_name, f_i, x_eff, B_eff):
    """
    Records the simulation data to a CSV file.

    Args:
        file_name: Path to the CSV file.
        f_i: Fraction parameter (e.g., f_g for generator failure).
        x_eff: System efficiency.
        B_eff: Calculated B_eff value.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    # Check if the file exists to write headers
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header
            writer.writerow(['f_i', 'x_eff', 'B_eff'])
        # Write data row
        writer.writerow([f_i, x_eff, B_eff])

# ============================
# Define Optimization Functions
# ============================

def optimize_power_distribution_refined(bipartite_graph, component_demands, generation):
    """
    Optimizes power distribution, prioritizing components with the least number of generators
    and generators with the least number of connections.

    Args:
        bipartite_graph: Dictionary representing the bipartite graph.
                         Keys are generators, values are lists of connected components.
        component_demands: List of total demands for each load component.
        generation: List of generator capacities.

    Returns:
        allocation: 2D list where allocation[i][j] is the power allocated from generator i to component j.
        total_power_delivered: Total power successfully delivered to all components.
    """
    num_generators = len(generation)
    num_components = len(component_demands)

    # Initialize allocation matrix
    allocation = [[0] * num_components for _ in range(num_generators)]

    # Remaining capacity of generators and demands of components
    remaining_gen = generation[:]
    remaining_demand = component_demands[:]

    # Step 1: Calculate generator availability for each component
    component_to_generators = {comp: [] for comp in range(num_components)}
    generator_to_components = {gen: [] for gen in range(num_generators)}

    for gen_index, gen_node in enumerate(bipartite_graph.keys()):
        for comp_index in bipartite_graph[gen_node]:
            component_to_generators[comp_index].append(gen_index)
            generator_to_components[gen_index].append(comp_index)

    # Step 2: Sort components by number of connected generators (ascending order)
    sorted_components = sorted(component_to_generators.keys(), key=lambda c: len(component_to_generators[c]))

    # Step 3: Allocate power prioritizing generator connections
    for comp_index in sorted_components:
        # Sort generators connected to this component by the number of other components they serve (ascending order)
        sorted_generators = sorted(component_to_generators[comp_index],
                                   key=lambda g: len(generator_to_components[g]))

        for gen_index in sorted_generators:
            if remaining_gen[gen_index] <= 0 or remaining_demand[comp_index] <= 0:
                continue

            # Allocate power
            power_to_transfer = min(remaining_gen[gen_index], remaining_demand[comp_index])
            allocation[gen_index][comp_index] += power_to_transfer
            remaining_gen[gen_index] -= power_to_transfer
            remaining_demand[comp_index] -= power_to_transfer

    # Calculate total power delivered
    total_power_delivered = sum(
        component_demands[i] - remaining_demand[i] for i in range(num_components)
    )

    return allocation, total_power_delivered

# ============================
# Define System State Function
# ============================

def system_state(adjacency_matrix, generators, generation, loads, demand):
    """
    Calculates the system's efficiency metrics (x_eff and B_eff).

    Args:
        adjacency_matrix: 2D list representing the adjacency matrix.
        generators: List of generator node indices.
        generation: List of generator capacities.
        loads: List of load node indices.
        demand: List of load demands.

    Returns:
        x_eff: System efficiency.
        B_eff: Effective efficiency metric as defined.
    """
    bipartite_graph, component_demands = create_bipartite_graph(adjacency_matrix, generators, loads, demand)
    
    # Print results
    print("Bipartite Graph:")
    for gen, comps in bipartite_graph.items():
        print(f"Generator {gen} -> Components {comps}")

    print("\nComponent Demands:")
    for i, comp_dem in enumerate(component_demands):
        print(f"Component {i}: Total Demand = {comp_dem}")

    # Optimize power distribution with refined prioritization
    allocation, total_power_delivered = optimize_power_distribution_refined(
        bipartite_graph, component_demands, generation
    )
    
    # Print results
    print("Power Allocation:")
    for i, row in enumerate(allocation):
        print(f"Generator {generators[i]}: {row}")

    print(f"\nTotal Power Delivered: {total_power_delivered}")

    print("\nRemaining Demands for Each Component:")
    for i, comp_dem in enumerate(component_demands):
        remaining = comp_dem - sum(row[i] for row in allocation)
        print(f"Component {i}: Remaining Demand = {remaining}")

    # Step 1: Calculate component received power
    component_received_power = [sum(allocation[gen][comp] for gen in range(len(generators))) for comp in range(len(component_demands))]

    # Step 2: Calculate load states using 'components'
    components, _ = load_demand_by_component(adjacency_matrix, generators, loads, demand)

    load_states = {}
    for comp_index, comp_load_set in enumerate(components):
        received_power = component_received_power[comp_index]
        total_demand = component_demands[comp_index]
        received_power_ratio = received_power / total_demand if total_demand > 0 else 0
        for load in comp_load_set:
            if load in loads:
                load_desire = demand[loads.index(load)]
                load_states[load] = load_desire * received_power_ratio
            else:
                print(f"Warning: Load {load} not found in loads.")

    # Step 3: Calculate x_eff
    x_eff_numerator = 0
    x_eff_denominator = len(loads)
    for load_node in loads:
        state = load_states.get(load_node, 0)
        outgoing_degree = sum(adjacency_matrix[load_node])
        x_eff_numerator += state * outgoing_degree

    x_eff = x_eff_numerator / x_eff_denominator if x_eff_denominator > 0 else 0

    # Step 4: Calculate B_eff
    B_eff = calculate_B_eff(adjacency_matrix, load_states)

    # Print efficiency metrics
    print(f"\nSystem Efficiency (x_eff): {x_eff}")
    print(f"System Efficiency (B_eff): {B_eff}")

    return x_eff, B_eff

# ============================
# Define Simulation Functions
# ============================

def simulate_line_failure(f_l, adjacency_matrix, generators, generation, loads, demand):
    # Convert adjacency_matrix to a NumPy array if it isn't already
    A = np.array(adjacency_matrix)
    
    # Identify all edges (nonzero entries)
    edges = [(i, j) for i in range(len(A)) for j in range(len(A)) if A[i][j] > 0]

    if not edges:
        print("No lines present to remove.")
        return system_state(A, generators, loads, demand)

    # Determine how many lines to remove
    total_lines = len(edges)
    lines_to_remove = int(f_l * total_lines)

    # Randomly choose lines to remove
    lines_removed = random.sample(edges, lines_to_remove)

    # Create a copy to avoid mutating the original matrix
    new_A = A.copy()
    for (i, j) in lines_removed:
        new_A[i][j] = 0

    # Run system state
    x_eff, B_eff = system_state(new_A, generators, generation, loads, demand)

    # Record data
    record_data(f"data/{SYSTEM_NAME}/line_failure_data.csv", f_l, x_eff, B_eff)

def simulate_generator_failure(f_g, adjacency_matrix, generators, generation, loads, demand):
    """
    Simulates generator failures by randomly removing a fraction f_g of generators.

    Args:
        f_g: Fraction of generators to fail [0,1]
        adjacency_matrix: 2D list representing the adjacency matrix
        generators: List of generator node indices
        generation: List of generator capacities
        loads: List of load node indices
        demand: List of load demands

    Returns:
        None
    """
    num_generators = len(generators)
    num_failures = int(f_g * num_generators)
    if num_failures == 0 and f_g > 0:
        num_failures = 1  # At least one failure if f_g > 0

    failed_gens = random.sample(generators, num_failures)
    print(f"\nSimulating Generator Failure: {failed_gens}")

    # Create a new adjacency matrix by removing all outgoing links from failed generators
    new_A = [row.copy() for row in adjacency_matrix]
    for gen in failed_gens:
        for j in range(len(new_A[gen])):
            new_A[gen][j] = 0  # Remove outgoing links

    # Set their generation to 0
    new_generation = generation.copy()
    for gen in failed_gens:
        gen_index = generators.index(gen)
        new_generation[gen_index] = 0

    # Run system state
    x_eff, B_eff = system_state(new_A, generators, new_generation, loads, demand)

    # Record data
    record_data(f"data/{SYSTEM_NAME}/generator_failure_data.csv", f_g, x_eff, B_eff)

def simulate_demand_increase(f_d, adjacency_matrix, generators, generation, loads, demand):
    """
    Simulates an increase in demand by a fraction f_d.

    Args:
        f_d: Fraction to increase demand [0,1]
        adjacency_matrix: 2D list representing the adjacency matrix
        generators: List of generator node indices
        generation: List of generator capacities
        loads: List of load node indices
        demand: List of load demands

    Returns:
        None
    """
    print(f"\nSimulating Demand Increase by {f_d*100}%")
    new_demand = [d * (1 + f_d) for d in demand]
    x_eff, B_eff = system_state(adjacency_matrix, generators, generation, loads, new_demand)
    record_data(f"data/{SYSTEM_NAME}/demand_increase_data.csv", f_d, x_eff, B_eff)

def simulate_generation_decrease(f_g, adjacency_matrix, generators, generation, loads, demand):
    """
    Simulates a decrease in generation capacity by a fraction f_g.

    Args:
        f_g: Fraction to decrease generation [0,1]
        adjacency_matrix: 2D list representing the adjacency matrix
        generators: List of generator node indices
        generation: List of generator capacities
        loads: List of load node indices
        demand: List of load demands

    Returns:
        None
    """
    print(f"\nSimulating Generation Decrease by {f_g*100}%")
    new_generation = [g * (1 - f_g) for g in generation]
    x_eff, B_eff = system_state(adjacency_matrix, generators, new_generation, loads, demand)
    record_data(f"data/{SYSTEM_NAME}/generation_decrease_data.csv", f_g, x_eff, B_eff)

def simulate_load_failure(f_l, adjacency_matrix, generators, generation, loads, demand):
    """
    Simulates load failures by randomly removing a fraction f_l of loads.

    Args:
        f_l: Fraction of loads to fail [0,1]
        adjacency_matrix: 2D list representing the adjacency matrix
        generators: List of generator node indices
        generation: List of generator capacities
        loads: List of load node indices
        demand: List of load demands

    Returns:
        None
    """
    num_loads = len(loads)
    num_failures = int(f_l * num_loads)
    if num_failures == 0 and f_l > 0:
        num_failures = 1  # At least one failure if f_l > 0

    failed_loads = random.sample(loads, num_failures)
    print(f"\nSimulating Load Failure: {failed_loads}")
    
    # Create a new adjacency matrix by removing all outgoing links from failed generators
    new_A = [row.copy() for row in adjacency_matrix]
    for load in failed_loads:
        for j in range(len(new_A[load])):
            new_A[load][j] = 0  # Remove outgoing links

    # Create a new demand list by setting failed loads' demand to 0
    new_demand = demand.copy()
    for load in failed_loads:
        load_index = loads.index(load)
        new_demand[load_index] = 0

    # Run system state
    x_eff, B_eff = system_state(new_A, generators, generation, loads, new_demand)

    # Record data
    record_data(f"data/{SYSTEM_NAME}/load_failure_data.csv", f_l, x_eff, B_eff)



# ============================
# Main Execution
# ============================

if __name__ == "__main__":
    # Load systems from JSON
    with open("systems_config.json", "r") as f:
        loaded_systems = json.load(f)

    # Example: Use BaseSystem for simulation
    selected_system = loaded_systems[SYSTEM_NAME]
    generators = selected_system["generators"]
    generation = selected_system["generation"]
    loads = selected_system["loads"]
    demand = selected_system["demand"]
    adjacency_matrix = selected_system["adjacency_matrix"]

    system_state(adjacency_matrix, generators, generation, loads, demand)
    simulate_line_failure(0.1, adjacency_matrix, generators, generation, loads, demand)
    system_state(adjacency_matrix, generators, generation, loads, demand)

    f_i_values = [round(i * 0.05, 2) for i in range(21)]  # [0.0, 0.05, 0.10, ..., 1.0]
    simulation_runs = 30
    for system_name in ['IEEE14','IEEE30','IEEE57','IEEE118']:
        SYSTEM_NAME = system_name
        for f_i in f_i_values:
            print(f"\n=== Simulation for f_i = {f_i} ===")
            for run in range(1, simulation_runs + 1):
                print(f"\n--- Run {run} for f_i = {f_i} ---")
                # Simulate each type of failure/increase/decrease
                simulate_line_failure(f_i, adjacency_matrix, generators, generation, loads, demand)
                simulate_generator_failure(f_i, adjacency_matrix, generators, generation, loads, demand)
                # simulate_demand_increase(f_i, adjacency_matrix, generators, generation, loads, demand)
                # simulate_generation_decrease(f_i, adjacency_matrix, generators, generation, loads, demand)
                simulate_load_failure(f_i, adjacency_matrix, generators, generation, loads, demand)

    print("\n=== All Simulations Completed ===")
