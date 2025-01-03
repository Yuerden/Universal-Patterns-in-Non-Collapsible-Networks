import json
import numpy as np

# Example: This is the base system you provided (7 nodes total: 2 generators, 5 loads)
# Let's call this a "small" system.
base_system = {
    "generators": [0, 1],
    "generation": [100, 100],  # Two generators
    "loads": [2, 3, 4, 5, 6],
    "demand": [70, 10, 25, 15, 50],  
    "adjacency_matrix": np.array([
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]
    ])
}

# For demonstration, we'll create three more systems with increasing number of nodes.
# In a real scenario, you would use the actual IEEE14, IEEE30, IEEE57, IEEE118 data.
# Here we just create placeholder data to show the concept.

def create_random_system(name, num_generators, num_loads):
    total_nodes = num_generators + num_loads
    gens = list(map(int, range(num_generators)))
    lds = list(map(int, range(num_generators, total_nodes)))

    # Convert generation to int
    gen_cap = [int(100) for _ in range(num_generators)]

    # Convert demands to a pure Python list of ints
    dem_array = np.random.randint(10, 101, size=num_loads)
    dem = [int(x) for x in dem_array]  # Ensures Python int, not np.int32

    # Create adjacency with pure Python ints
    adjacency = (np.random.rand(total_nodes, total_nodes) < 0.2).astype(int)
    np.fill_diagonal(adjacency, 0)
    adjacency_list = [list(map(int, row)) for row in adjacency]  # Map each element to int

    return {
        "generators": gens,
        "generation": gen_cap,
        "loads": lds,
        "demand": dem,
        "adjacency_matrix": adjacency_list
    }


# Create three more systems (just examples with increasing complexity)
# For example:
# IEEE14: Let's say 4 generators, 10 loads
ieee14 = create_random_system("IEEE14", 4, 10)

# IEEE30: 6 generators, 24 loads
ieee30 = create_random_system("IEEE30", 6, 24)

# IEEE57: 7 generators, 50 loads
ieee57 = create_random_system("IEEE57", 7, 50)

# IEEE118: 10 generators, 108 loads
ieee118 = create_random_system("IEEE118", 10, 108)

# Now we combine all into one dictionary
systems = {
    "BaseSystem": {
        "generators": base_system["generators"],
        "generation": base_system["generation"],
        "loads": base_system["loads"],
        "demand": base_system["demand"],
        # BaseSystem adjacency_matrix is still a numpy array, so we convert it
        "adjacency_matrix": base_system["adjacency_matrix"].astype(int).tolist()
    },
    "IEEE14": {
        "generators": ieee14["generators"],
        "generation": ieee14["generation"],
        "loads": ieee14["loads"],
        "demand": ieee14["demand"],
        # ieee14["adjacency_matrix"] is already a list of ints
        "adjacency_matrix": ieee14["adjacency_matrix"]
    },
    "IEEE30": {
        "generators": ieee30["generators"],
        "generation": ieee30["generation"],
        "loads": ieee30["loads"],
        "demand": ieee30["demand"],
        "adjacency_matrix": ieee30["adjacency_matrix"]
    },
    "IEEE57": {
        "generators": ieee57["generators"],
        "generation": ieee57["generation"],
        "loads": ieee57["loads"],
        "demand": ieee57["demand"],
        "adjacency_matrix": ieee57["adjacency_matrix"]
    },
    "IEEE118": {
        "generators": ieee118["generators"],
        "generation": ieee118["generation"],
        "loads": ieee118["loads"],
        "demand": ieee118["demand"],
        "adjacency_matrix": ieee118["adjacency_matrix"]
    }
}

with open("systems_config.json", "w") as f:
    json.dump(systems, f, indent=4)