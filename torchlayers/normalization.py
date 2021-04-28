import collections

import torch

from . import module


class InstanceNorm(module.InferDimension):
    """Apply Instance Normalization over inferred dimension (3D up to 5D).

    Based on input shape it either creates 1D, 2D or 3D instance normalization for inputs of shape
    3D, 4D, 5D respectively (including batch as first dimension).

    Otherwise works like standard PyTorch's `InstanceNorm [here](https://pytorch.org/docs/stable/nn.html#torch.nn.InstanceNorm1d)

    Attributes:
        num_features :
            :math:`C` (number of channels in input) from an expected input.
            Can be number of outputs of previous linear layer as well
        eps :
            Value added to the denominator for numerical stability.
            Default: `1e-5`
        momentum :
            Value used for the `running_mean` and `running_var`
            computation. Default: `0.1`
        affine :
            If ``True``, this module has learnable affine parameters, initialized just like in batch normalization.
            Default: ``False``
        track_running_stats :
            If ``True``, this module tracks the running mean and variance,
            and when set to ``False``, this module does not track such statistics and always uses batch
            statistics in both training and eval modes.
            Default: ``False``

    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        """Initialize `InstanceNorm` object.
        
        Arguments:
            num_features :
                :math:`C` (number of channels in input) from an expected input.
                Can be number of outputs of previous linear layer as well
            eps :
                Value added to the denominator for numerical stability.
                Default: `1e-5`
            momentum :
                Value used for the `running_mean` and `running_var`
                computation. Default: `0.1`
            affine :
                If ``True``, this module has learnable affine parameters, initialized just like in batch normalization.
                Default: ``False``
            track_running_stats :
                If ``True``, this module tracks the running mean and variance,
                and when set to ``False``, this module does not track such statistics and always uses batch
                statistics in both training and eval modes.
                Default: ``False``
        """
        super().__init__(
            dispatcher={
                5: torch.nn.InstanceNorm3d,
                4: torch.nn.InstanceNorm2d,
                3: torch.nn.InstanceNorm1d,
            },
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


class BatchNorm(module.InferDimension):
    """Apply Batch Normalization over inferred dimension (2D up to 5D).

    Based on input shape it either creates `1D`, `2D` or `3D` batch normalization for inputs of shape
    `2D/3D`, `4D`, `5D` respectively (including batch as first dimension).

    Otherwise works like standard PyTorch's `BatchNorm [here](https://pytorch.org/docs/stable/nn.html#batchnorm1d)

    Attributes:
        num_features :
            :math:`C` (number of channels in input) from an expected input.
            Can be number of outputs of previous linear layer as well
        eps :
            Value added to the denominator for numerical stability.
            Default: `1e-5`
        momentum :
            Value used for the `running_mean` and `running_var`
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: `0.1`
        affine :
            If ``True``, this module has learnable affine parameters.
            Default: ``True``
        track_running_stats :
            If ``True``, this module tracks the running mean and variance,
            and when set to ``False``, this module does not track such statistics and always uses batch
            statistics in both training and eval modes.
            Default: ``True``

    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """Initialize `BatchNorm` object.
        
        Arguments:
            num_features :
                :math:`C` (number of channels in input) from an expected input.
                Can be number of outputs of previous linear layer as well
            eps :
                Value added to the denominator for numerical stability.
                Default: `1e-5`
            momentum :
                Value used for the `running_mean` and `running_var`
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: `0.1`
            affine :
                If ``True``, this module has learnable affine parameters.
                Default: ``True``
            track_running_stats :
                If ``True``, this module tracks the running mean and variance,
                and when set to ``False``, this module does not track such statistics and always uses batch
                statistics in both training and eval modes.
                Default: ``True``
        """
        super().__init__(
            dispatcher={
                5: torch.nn.BatchNorm3d,
                4: torch.nn.BatchNorm2d,
                3: torch.nn.BatchNorm1d,
                2: torch.nn.BatchNorm1d,
            },
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


# Arguments had the wrong order unfortunately
class GroupNorm(torch.nn.GroupNorm):
    """Apply Group Normalization over a mini-batch of inputs.

    Works exactly like PyTorch's counterpart, but `num_channels` is used as first argument
    so it can be inferred during first forward pass.

    Attributes:
        num_channels :
            Number of channels expected in input
        num_groups :
            Number of groups to separate the channels into
        eps :
            Value added to the denominator for numerical stability.
            Default: `1e-5`
        affine :
            If ``True``, this module has learnable affine parameters.
            Default: ``True``

    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-05,
        affine: bool = True,
    ):
        """Initialize `GroupNorm` object.
        
        Arguments:
            num_channels :
                Number of channels expected in input
            num_groups :
                Number of groups to separate the channels into
            eps :
                Value added to the denominator for numerical stability.
                Default: `1e-5`
            affine :
                If ``True``, this module has learnable affine parameters.
                Default: ``True``
        """
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine,
        )
