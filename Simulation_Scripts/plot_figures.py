import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_simulation_data(data_dir):
    """
    Reads all CSV files from the data directory and aggregates them into a nested dictionary.
    
    Args:
        data_dir (str): Path to the data directory containing system model folders.
    
    Returns:
        dict: Nested dictionary with structure data[system_model][simulation_type] = DataFrame
    """
    data = {}
    simulation_types = ['line_failure', 'generator_failure', 'load_failure']
    
    # Traverse through each system model directory
    for system_model in os.listdir(data_dir):
        system_path = os.path.join(data_dir, system_model)
        if not os.path.isdir(system_path):
            continue  # Skip files, only process directories
        
        data[system_model] = {}
        
        # Traverse through each simulation type
        for sim_type in simulation_types:
            file_name = f"{sim_type}_data.csv"
            file_path = os.path.join(system_path, file_name)
            if not os.path.isfile(file_path):
                print(f"Warning: {file_path} does not exist.")
                continue  # Skip missing files
            
            # Read CSV into DataFrame
            try:
                df = pd.read_csv(file_path)
                data[system_model][sim_type] = df
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return data

def plot_x_eff_vs_f_i(system_model, simulations):
    """
    Plots x_eff vs f_i for each simulation type of a given system.
    
    Args:
        system_model (str): Name of the system model (e.g., "IEEE14")
        simulations (dict): Dictionary of simulation_type -> DataFrame
    """
    plt.figure(figsize=(18, 6))
    sns.set(style="whitegrid")
    
    simulation_types = ['line_failure', 'generator_failure', 'load_failure']
    sim_titles = {
        'line_failure': 'Line Failure',
        'generator_failure': 'Generator Failure',
        'load_failure': 'Load Failure'
    }
    markers = {
        'line_failure': 'o',
        'generator_failure': 's',
        'load_failure': '^'
    }
    colors = sns.color_palette("husl", len(simulation_types))
    
    for idx, sim_type in enumerate(simulation_types):
        ax = plt.subplot(1, 3, idx+1)
        if sim_type not in simulations:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.set_title(f"{sim_titles[sim_type]}")
            ax.set_xlabel('Failure Fraction (f_i)')
            ax.set_ylabel('System Efficiency (x_eff)')
            continue
        df = simulations[sim_type]
        # Aggregate data: compute mean and std for x_eff at each f_i
        agg_df = df.groupby('f_i').agg(mean_x_eff=('x_eff', 'mean'), std_x_eff=('x_eff', 'std')).reset_index()
        
        sns.lineplot(
            data=agg_df,
            x='f_i',
            y='mean_x_eff',
            marker=markers[sim_type],
            label=sim_titles[sim_type],
            color=colors[idx],
            ax=ax
        )
        # Add error bars
        ax.fill_between(
            agg_df['f_i'],
            agg_df['mean_x_eff'] - agg_df['std_x_eff'],
            agg_df['mean_x_eff'] + agg_df['std_x_eff'],
            alpha=0.2,
            color=colors[idx]
        )
        ax.set_title(f"{sim_titles[sim_type]}")
        ax.set_xlabel('Failure Fraction (f_i)')
        ax.set_ylabel('System Efficiency (x_eff)')
    
    plt.suptitle(f'{system_model} - x_eff vs. f_i for Different Simulation Types', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_combined_x_eff_vs_B_eff(system_model, simulations):
    """
    Plots combined x_eff vs B_eff for all simulation types of a given system.
    
    Args:
        system_model (str): Name of the system model (e.g., "IEEE14")
        simulations (dict): Dictionary of simulation_type -> DataFrame
    """
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    
    simulation_types = ['line_failure', 'generator_failure', 'load_failure']
    sim_titles = {
        'line_failure': 'Line Failure',
        'generator_failure': 'Generator Failure',
        'load_failure': 'Load Failure'
    }
    markers = {
        'line_failure': 'o',
        'generator_failure': 's',
        'load_failure': '^'
    }
    colors = sns.color_palette("husl", len(simulation_types))
    
    for idx, sim_type in enumerate(simulation_types):
        if sim_type not in simulations:
            continue
        df = simulations[sim_type]
        sns.scatterplot(
            data=df,
            x='B_eff',
            y='x_eff',
            label=sim_titles[sim_type],
            marker=markers[sim_type],
            color=colors[idx],
            s=100,
            edgecolor='k',
            alpha=0.7
        )
    
    plt.title(f'{system_model} - x_eff vs. B_eff Across Simulation Types', fontsize=16)
    plt.xlabel('Effective Efficiency (B_eff)', fontsize=14)
    plt.ylabel('System Efficiency (x_eff)', fontsize=14)
    plt.legend(title='Simulation Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    data_directory = "./data"
    
    # Read and aggregate simulation data
    simulation_data = read_simulation_data(data_directory)
    
    if not simulation_data:
        print("No simulation data found. Please ensure that CSV files are present in the data directory.")
        return
    
    # List of IEEE systems to plot
    ieee_systems = ['IEEE14', 'IEEE30', 'IEEE57', 'IEEE118']
    
    for system_model in ieee_systems:
        if system_model not in simulation_data:
            print(f"Warning: No data found for system model '{system_model}'. Skipping plots for this system.")
            continue
        simulations = simulation_data[system_model]
        
        # Plot x_eff vs f_i for each simulation type
        plot_x_eff_vs_f_i(system_model, simulations)
        
        # Plot combined x_eff vs B_eff
        plot_combined_x_eff_vs_B_eff(system_model, simulations)
    
    print("\n=== All Plots Generated Successfully ===")

if __name__ == "__main__":
    main()
