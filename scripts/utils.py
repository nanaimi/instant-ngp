import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt


def get_df_runs(runs: wandb.apis.public.Runs) -> pd.dataframe:
    run_data = []
    run_configs = []
    for i, run in enumerate(runs):
        # Extract the configuration as a dictionary
        run_config = run.config
        if len(run_config) < 10:
            continue
        # print(len(run_config))

        # Fetch the run's data as a pandas dataframe
        run_dataframe = run.history().dropna()
        run_shape = run_dataframe.shape
        if run_shape[0] == 0:
            continue
        run_dataframe["run_name"] = run.name
        run_dataframe["n_levels"] = run_config['encoding']['n_levels']
        run_dataframe["base_resolution"] = run_config['encoding']['base_resolution']
        run_dataframe["log2_hashmap_size"] = run_config['encoding']['log2_hashmap_size']
        run_dataframe["per_level_scale"] = run_config['encoding']['per_level_scale']
        run_dataframe["n_features_per_level"] = run_config['encoding']['n_features_per_level']
        run_dataframe["n_enc_params"] = run_config['n_enc_params']
        run_dataframe["n_params"] = run_config['n_params']

        # Append the run's data to the list
        run_data.append(run_dataframe)
    
    aggregated_df = pd.concat(run_data, axis=0, ignore_index=True)
    return aggregated_df

def estimated_bpp(n_params: np.ndarray) -> np.ndarray:
    return (n_params * 8) / (512 * 768)

def psnr_yield_per_param(psnr: np.ndarray, n_enc_params: np.ndarray) -> : np.ndarray:
    return psnr / n_enc_params

def hypotheical_bpp_boundary(n_hidden_layers: int=2, n_neurons: int=64, width: int=512, height: int=768):
    nr_mlp_params = (n_hidden_layers - 1) * (n_neurons ** 2) + n_neurons * 3 + 2 * n_neurons
    return (nr_mlp_params * 8) / (width * height)


def plot_mult_dfs(dfs: pd.dataframe, save: bool=True, estimated_bpp: bool=True):
    # Color palette
    color_palette = sns.color_palette('Set1')  # Palettes: 'Set1', 'Set2', 'Dark2', 'Paired', etc.

    # Select the two columns for the scatter plot
    plt.figure(figsize=(16, 8))
    
    # Set axis ranges
    plt.xlim(0, 10) 
    plt.ylim(15, 50)

    i = 0
    for data, label in zip(agg_data, labels):
        sorted_data = data.sort_values(by='estimated_bpp', ascending=True) if estimated_bpp else data.sort_values(by='n_params', ascending=True)
        x_data = sorted_data['estimated_bpp'] if estimated_bpp else sorted_data['n_params']
        y_data = sorted_data['psnr']
        plt.plot(x_data, y_data, color=color_palette[i], label=f"""{label}""", marker=next(marker_styles), linestyle=next(line_styles))
        i += 1

    # Add labels and title
    x_label = 'estimated bpp' if estimated_bpp else 'n_params'
    plt.xlabel(f"""{x_label}""")
    plt.ylabel('PSNR')
    plt.title('Encoding Parameter Scaling')
    
    plt.legend()
    if save: 
        plt.savefig('parameter_scaling_bpp_psnr.png')
    # Show the plot
    plt.show()


def plot_psnr_param(param: str, data_df: pd.dataframe, save: bool=True):
    plt.figure(figsize=(16, 8))

    # axis ranges
    # plt.xlim(0, 1)  # Set the minimum and maximum values for the x-axis
    # plt.ylim(14, 45)  # Set the minimum and maximum values for the x-axis
    
    sorted_data = data_df.sort_values(by=f"""{param}""", ascending=True)
    x_data = sorted_data[f"""{param}"""]
    y_data = sorted_data['psnr']
    
    # Create the scatter plot
    plt.scatter(x_data, y_data)

    # Add labels and title
    plt.xlabel(f"""{param}""")
    plt.ylabel('PSNR')
    plt.title(f"""PSNR v. {param}""")

    # Show the plot
    if save:
        plt.savefig(f"""{param}_psnr.png""")

    plt.show()