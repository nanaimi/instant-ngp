# import itertools
from typing import Tuple

# import jax
# import jax.numpy as jnp
import numpy as np

MAX_BYTE = 2.0**8 - 1.0  # 255.0
PROCESSING_DTYPE = np.float32


# Implement uniform, asymmetric, uint8 bit quantization
def quantization_uniform_asymmetric(
    x: np.ndarray, bits: int = 8
) -> Tuple[np.uint16, np.float32, np.float32, np.dtype]:
    """Uniformly quantizes x to bits number of bits in asymmetric mode.
    Args:
        x: Input array.
        bits: Number of bits to quantize to.
    Returns:
        x_q: Quantized array.
        q_x: Quantization factor.
        zpx: Zero-point.
        dtype: original dtype of x.
    Raises:
        ValueError: If bits is not between 1 and 16. Otherwise quantization overflows.
    """
    x_dtype = x.dtype
    if bits > 16 or bits < 1:
        raise ValueError("nr of bits for quantization must be between 1 and 16")
    MAX_NUM = np.add(2.0**bits, -1.0, dtype=PROCESSING_DTYPE)
    x_min, x_max = x.min(), x.max()
    q_x = MAX_NUM / (x_max - x_min)
    zpx = x_min * q_x
    # zpx = np.round(x_min * q_x)
    x_q = np.round(q_x * x.astype(PROCESSING_DTYPE) - zpx).astype(np.uint16)
    return x_q, q_x, zpx, x_dtype


def dequantize_uniform_asymmetric(
    x_q: np.ndarray, q_x: np.ndarray, zpx: np.ndarray, dtype: np.dtype
) -> np.ndarray:
    """Dequantizes x_q to float32.
    Args:
        x_q: Quantized array.
        q_x: Quantization factor.
        zpx: Zero-point.
    Returns:
        x: Dequantized array.
    """
    x_dq = np.add(x_q, zpx, dtype=PROCESSING_DTYPE) / q_x
    x_dq = x_dq.astype(dtype)
    return x_dq


# Taken from:
# https://github.com/google-research/google-research/blob/master/merf/internal/quantize.py


# def differentiable_byte_quantize(x):
#     """Implements rounding with a straight-through-estimator."""
#     zero = x - jax.lax.stop_gradient(x)
#     return zero + jax.lax.stop_gradient(
#         jnp.round(jnp.clip(x, 0.0, 1.0) * MAX_BYTE) / MAX_BYTE
#     )


# def simulate_quantization(x, v_min, v_max):
#     """Simulates quant. during training: [-inf, inf] -> [v_min, v_max]."""
#     x = jax.nn.sigmoid(x)  # Bounded to [0, 1].
#     x = differentiable_byte_quantize(x)  # quantize and dequantize.
#     return math.denormalize(x, v_min, v_max)  # Bounded to [v_min, v_max].


# def dequantize_and_interpolate(x_grid, data, v_min, v_max):
#     """Dequantizes and denormalizes and then linearly interpolates grid values."""
#     x_floor = jnp.floor(x_grid).astype(jnp.int32)
#     x_ceil = jnp.ceil(x_grid).astype(jnp.int32)
#     local_coordinates = x_grid - x_floor
#     res = jnp.zeros(x_grid.shape[:-1] + (data.shape[-1],))
#     corner_coords = [[False, True] for _ in range(local_coordinates.shape[-1])]
#     for z in itertools.product(*corner_coords):
#         w = jnp.ones(local_coordinates.shape[:-1])
#         l = []  # noqa
#         for i, b in enumerate(z):
#             w = w * (
#                 local_coordinates[Ellipsis, i]
#                 if b
#                 else (1 - local_coordinates[Ellipsis, i])
#             )
#             l.append(x_ceil[Ellipsis, i] if b else x_floor[Ellipsis, i])
#         gathered_data = data[tuple(l)]
#         gathered_data = dequantize_byte_to_float(gathered_data, jnp)
#         gathered_data = math.denormalize(gathered_data, v_min, v_max)
#         res = res + w[Ellipsis, None] * gathered_data.reshape(res.shape)
#     return res


# def map_quantize(*l):  # noqa
#     """For quantization after training."""

#     def sigmoid_and_quantize_float_to_byte(x):
#         if x is None:
#             return None
#         cpu = jax.devices("cpu")[0]  # Prevents JAX from moving array to GPU.
#         x = jax.device_put(x, cpu)
#         x = jax.nn.sigmoid(x)
#         return quantize_float_to_byte(x)

#     return jax.tree_map(sigmoid_and_quantize_float_to_byte, l)
