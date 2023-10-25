import torch
import copy


def assert_input(input):
    if not torch.is_tensor(input):
        copy_input = copy.deepcopy(input)
        input = torch.from_numpy(copy_input)
    if input.shape[-1] != 3:
        raise AssertionError(
            f"The length of the third dimension ({input.shape[-1]}) does not match the expected length ({3})"
        )
    return input
