import torch

from models.data_utils import CONSTANTS

# @perf_compute.log
def replace_constant(yhat, out_variables):
    for i in range(yhat.shape[1]):
        # if constant replace with 0.0
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = 0.0
    return yhat


# @perf_compute.log
def pad(patch_size: int, x: torch.Tensor) -> tuple[torch.Tensor, int]:
    h = x.shape[-2]
    # Calculate the pad size for the height if it's not divisible by the patch size
    if h % patch_size != 0:
        pad_size = patch_size - h % patch_size
        # Padding format (left, right, top, bottom) for the last two dimensions, 0s for the rest
        # Since we only want to pad the top, we set it as (0, 0, pad_size, 0)
        padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), "constant", 0)
    else:
        padded_x = x
        pad_size = 0
    return padded_x, pad_size


__all__ = ["replace_constant", "pad"]
