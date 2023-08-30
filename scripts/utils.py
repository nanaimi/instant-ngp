import math
import os
import sys
import zlib
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import msgpack
import numpy as np
import pandas as pd
import wandb

sys.path.append(os.path.abspath("/Users/nasib/code/instant-ngp/scripts"))
from scenes import *  # noqa

T_co = TypeVar("T_co", bound=np.generic, covariant=True)

Vector = np.ndarray[Tuple[int], np.dtype[T_co]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[T_co]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T_co]]


def get_df_runs(runs: wandb.apis.public.Runs) -> pd.DataFrame:
    """Get the dataframe of runs.

    Args:
        runs (wandb.apis.public.Runs): wandb runs.

    Returns:
        pd.DataFrame: dataframe of runs.
    """
    run_data: List[pd.DataFrame] = []
    for _, run in enumerate(runs):
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
        run_dataframe["n_levels"] = run_config["encoding"]["n_levels"]
        run_dataframe["base_resolution"] = run_config["encoding"]["base_resolution"]
        run_dataframe["log2_hashmap_size"] = run_config["encoding"]["log2_hashmap_size"]
        run_dataframe["per_level_scale"] = run_config["encoding"]["per_level_scale"]
        run_dataframe["n_features_per_level"] = run_config["encoding"][
            "n_features_per_level"
        ]
        run_dataframe["n_enc_params"] = run_config["n_enc_params"]
        run_dataframe["n_params"] = run_config["n_params"]
        run_dataframe["estimated_bpp"] = estimated_bpp(
            run_dataframe["n_params"], 512, 768
        )
        run_dataframe["psnr_yield_per_param"] = psnr_yield_per_param(
            run_dataframe["psnr"], run_dataframe["n_enc_params"]
        )

        columns_to_select = [
            "run_name",
            "n_levels",
            "base_resolution",
            "log2_hashmap_size",
            "per_level_scale",
            "n_features_per_level",
            "n_enc_params",
            "n_params",
            "loss",
            "psnr",
            "estimated_bpp",
            "psnr_yield_per_param",
        ]
        selected_columns_df = run_dataframe[columns_to_select]

        # Append the run's data to the list
        run_data.append(selected_columns_df)

    aggregated_df = pd.concat(run_data, axis=0, ignore_index=True)
    return aggregated_df


def estimated_bpp(n_params: int, pixel_width: int, pixel_height: int) -> np.float32:
    """Naive estimate of the bits per pixel or rate. Assumes 8 bits per parameter.

    Args:
        n_params (int): the number of parameters.

    Returns:
        np.float32: estimated bits per pixel.
    """
    return np.float32((n_params * 8) / (pixel_width * pixel_height))


def psnr_yield_per_param(
    psnr: Vector[np.float32], n_enc_params: Vector[np.float32]
) -> Vector[np.float32]:
    """Compute the PSNR yield per parameter.

    Args:
        psnr (Vector[np.float32]): vector of PSNR values.
        n_enc_params (Vector[np.float32]): vector of number of encoding parameters.

    Returns:
        Vector[np.float32]: vector of PSNR yield per parameter.
    """
    assert (
        psnr.shape == n_enc_params.shape
    ), "Shape mismatch, cannot compute yield per param"
    return psnr / n_enc_params


def hypothetical_bpp_boundary(
    n_hidden_layers: int = 2, n_neurons: int = 64, width: int = 512, height: int = 768
):
    """Compute the hypothetical bpp boundary for a given MLP.

    Args:
        n_hidden_layers (int, optional): number of hidden layers. Defaults to 2.
        n_neurons (int, optional): number of neurons per hidden layer. Defaults to 64.
        width (int, optional): width of the image. Defaults to 512.
        height (int, optional): height of the image. Defaults to 768.

    Returns:
        float: hypothetical bpp boundary.
    """
    nr_mlp_params = (
        (n_hidden_layers - 1) * (n_neurons**2) + n_neurons * 3 + 2 * n_neurons
    )
    return (nr_mlp_params * 8) / (width * height)


# def hypotheical_bpp_boundary(
#     n_hidden_layers: int = 2, n_neurons: int = 64, width: int = 512, height: int = 768
# ):
#     nr_mlp_params = (
#         (n_hidden_layers - 1) * (n_neurons**2) + n_neurons * 3 + 2 * n_neurons
#     )
#     return (nr_mlp_params * 8) / (width * height)


# Encoding parameter calculations
def compute_scaling_parameter(
    max_resolution, min_resolution, n_levels: int
) -> np.float32:
    """Computes the scaling parameter for a given encoding scheme.

    Args:
        max_resolution (int): maximum resolution of the grid.
        min_resolution (int): minimum resolution of the grid.
        n_levels (int): number of levels in the encoding scheme.

    Returns:
        float: scaling parameter.
    """
    return np.exp((np.log(max_resolution) - math.log(min_resolution)) / (n_levels - 1))


def encoding_parameter_calculator(
    n_levels: int,
    n_features: int,
    hashtable_size: int,
    min_resolution: int,
    max_resolution: Optional[int] = None,
    scaling_parameter: Optional[np.float32] = None,
):
    """Computes the number of encoding parameters for a given encoding scheme.

    Args:
        n_levels (int): number of levels in the encoding scheme.
        n_features (int): number of features per level.
        hashtable_size (int): size of the hashtable.
        min_resolution (int): minimum resolution of the grid.
        max_resolution (int, optional): maximum resolution of the grid. Defaults to None.
        scaling_parameter (float, optional): scaling parameter. Defaults to None.

    Returns:
        int: number of encoding parameters.
    """
    total = 0
    if scaling_parameter is None and max_resolution is not None:
        scaling_parameter = compute_scaling_parameter(
            max_resolution, min_resolution, n_levels
        )

    if scaling_parameter is None:
        raise ValueError("scaling_parameter and max_resolution cannot both be None")

    for level in range(n_levels):
        grid_res = np.floor(min_resolution * (scaling_parameter**level))
        grid_size = grid_res**2
        total += min(grid_size, hashtable_size) * n_features
    return total


def get_list_of_grid_resolutions(
    min_resolution: int, scaling_parameter: np.float32, n_levels: int
):
    """Returns a list of grid resolutions for a given encoding scheme.

    Args:
        min_resolution (int): minimum resolution of the grid.
        scaling_parameter (float): scaling parameter.
        n_levels (int): number of levels in the encoding scheme.

    Returns:
        List[int]: list of grid resolutions.
    """
    return [
        int(np.floor(min_resolution * (scaling_parameter**level)))
        for level in range(n_levels)
    ]


def get_scene(scene):
    """Returns the scene object for the given scene name."""
    for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:  # noqa
        if scene in scenes:
            return scenes[scene]
    return None


# Plotting Functions
# def plot_mult_dfs(dfs: pd.DataFrame, save: bool = True, estimated_bpp: bool = True):
#     # Color palette
#     color_palette = sns.color_palette(
#         "Set1"
#     )  # Palettes: 'Set1', 'Set2', 'Dark2', 'Paired', etc.

#     # Select the two columns for the scatter plot
#     plt.figure(figsize=(16, 8))

#     # Set axis ranges
#     plt.xlim(0, 10)
#     plt.ylim(15, 50)

#     i = 0
#     for data, label in zip(agg_data, labels):
#         sorted_data = (
#             data.sort_values(by="estimated_bpp", ascending=True)
#             if estimated_bpp
#             else data.sort_values(by="n_params", ascending=True)
#         )
#         x_data = (
#             sorted_data["estimated_bpp"] if estimated_bpp else sorted_data["n_params"]
#         )
#         y_data = sorted_data["psnr"]
#         plt.plot(
#             x_data,
#             y_data,
#             color=color_palette[i],
#             label=f"""{label}""",
#             marker=next(marker_styles),
#             linestyle=next(line_styles),
#         )
#         i += 1

#     # Add labels and title
#     x_label = "estimated bpp" if estimated_bpp else "n_params"
#     plt.xlabel(f"""{x_label}""")
#     plt.ylabel("PSNR")
#     plt.title("Encoding Parameter Scaling")

#     plt.legend()
#     if save:
#         plt.savefig("parameter_scaling_bpp_psnr.png")
#     # Show the plot
#     plt.show()


def plot_psnr_param(param: str, data_df: pd.DataFrame, save: bool = True):
    """Plot the PSNR v. parameter.

    Args:
        param (str): parameter to plot.
        data_df (pd.DataFrame): dataframe containing the data.
        save (bool, optional): whether to save the plot. Defaults to True.
    """
    plt.figure(figsize=(16, 8))

    # axis ranges
    # plt.xlim(0, 1)  # Set the minimum and maximum values for the x-axis
    # plt.ylim(14, 45)  # Set the minimum and maximum values for the x-axis

    sorted_data = data_df.sort_values(by=f"""{param}""", ascending=True)
    x_data = sorted_data[f"""{param}"""]
    y_data = sorted_data["psnr"]

    # Create the scatter plot
    plt.scatter(x_data, y_data)

    # Add labels and title
    plt.xlabel(f"""{param}""")
    plt.ylabel("PSNR")
    plt.title(f"""PSNR v. {param}""")

    # Show the plot
    if save:
        plt.savefig(f"""{param}_psnr.png""")

    plt.show()


def plot_quantization_bins(x_q: np.ndarray, n_bins: int = 256):
    """Plot the histogram of quantization bins.

    Args:
        x_q (np.ndarray): quantized array.
        bins (int, optional): number of bins. Defaults to 256.
    """
    # Calculate the histogram
    bin_range = (0, n_bins)
    hist, bins = np.histogram(x_q, bins=n_bins, range=bin_range)

    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist, width=2, align="center")
    plt.xlabel("Quantization Bins")
    plt.ylabel("Frequency")
    plt.title("Histogram of Quantization Bins")
    plt.show()


def plot_flattened_params(param_vector: np.ndarray):
    """Plot the flateened parameters in index order.

    Args:
        param_vector (np.ndarray): flattened parameter vector.
    """
    indices = np.arange(len(param_vector))

    plt.figure(figsize=(10, 6))
    plt.plot(indices, param_vector, marker="o", markersize=2, linestyle="--", color="b")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Flattened Array Plot")
    plt.grid(True)
    plt.show()


def statistics_of_params(param_vector: np.ndarray) -> Tuple[float, ...]:
    """Retrieve statistics of the flattened parameters.

    Args:
        param_vector (np.ndarray): flattened parameter vector.

    Returns:
        Tuple[float, ...]: tuple of statistics (min, max, std, mean, median).
    """
    x_min = param_vector.min()
    x_max = param_vector.max()
    x_std = param_vector.std()
    x_mean = param_vector.mean()
    x_median = np.median(param_vector)

    return x_min, x_max, x_std, x_mean, x_median


def compress_and_serialize_ingp(
    snapshot: Dict[str, Any],
    compress: bool = True,
    compression_level: int = zlib.Z_DEFAULT_COMPRESSION,
) -> bytes:
    """Compress and serialize to .ingp snapshot.

    Args:
        snapshot (Dict[str, Any]): snapshot to serialize.
        compress (bool, optional): whether to compress the snapshot. Defaults to True.
        compression_level (int, optional): compression level. Defaults to zlib default.

    Returns:
        bytes: serialized snapshot.
    """
    serialized_snapshot = msgpack.packb(snapshot, use_bin_type=True)
    packed_snapshot = serialized_snapshot
    if compress:
        compressed_snapshot = zlib.compress(
            serialized_snapshot, level=compression_level
        )
        packed_snapshot = compressed_snapshot
    return packed_snapshot


def deserialize_ingp(snapshot_path: str) -> Dict[str, Any]:
    """Deserialize a snapshot file.

    Args:
        snapshot_path (str): path to the snapshot file.

    Returns:
        Dict[str, Any]: deserialized snapshot.
    """
    with open(snapshot_path, "rb", buffering=0) as snapshot_file:
        # Read the zlib-compressed data.
        compressed_data = snapshot_file.read()
        # Decompress the zlib-compressed data. For most typical use cases, use the
        # default value for wbits (MAX_WBITS | 16 (or simply -15)). This setting tells
        # zlib to automatically detect the window size based on the zlib or gzip header
        # present in the compressed data.
        decompressed_data = zlib.decompress(compressed_data, wbits=zlib.MAX_WBITS | 16)
        # Deserialize the MessagePack data.
        result = msgpack.unpackb(decompressed_data, raw=False)
        return result


def load_parameters_from_snapshot(
    snapshot_path: str,
) -> Tuple[Vector[np.float32], Dict[str, Any]]:
    """Load the parameters from a snapshot file.

    Args:
        snapshot_path (str): path to the snapshot file.

    Returns:
        Dict[str, Any]: deserialized snapshot.
    """
    snapshot = deserialize_ingp(snapshot_path)

    if type in ["float", "__half"]:
        np_dtype = np.float32 if type == "float" else np.float16

        params_array = np.frombuffer(
            snapshot["snapshot"]["params_binary"], dtype=np_dtype
        )
        if type == "__half":
            params_half = np.array(params_array, dtype=np.float16)
            params_fp = np.array(params_array, dtype=np.float32)
            assert np.allclose(
                params_half, params_fp, atol=1e-10
            ), "parameters mismatch"
        else:
            params_fp = np.array(params_array, dtype=np.float32)
    else:
        raise RuntimeError(
            "Trainer: snapshot parameters must be of precision float or __half"
        )

    return params_fp, snapshot
