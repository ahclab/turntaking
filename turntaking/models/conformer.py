from pprint import pprint
import torch
import torch.nn as nn
import math

from typing import Optional, Tuple

from turntaking.models.multi_head_attention import (
    MultiHeadAttentionAlibi,
    MultiHeadAttention,
)

def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask

class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class StaticPositionEmbedding(nn.Module):
    def __init__(self, seqlen, dmodel):
        super(StaticPositionEmbedding, self).__init__()
        pos = torch.arange(0.0, seqlen).unsqueeze(1).repeat(1, dmodel)
        dim = torch.arange(0.0, dmodel).unsqueeze(0).repeat(seqlen, 1)
        div = torch.exp(
            -math.log(10000) * (2 * torch.div(dim, 2, rounding_mode="trunc") / dmodel)
        )
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer("pe", pos.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]

class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=False),
            torch.nn.SiLU(),
            # torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=False),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)

class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.1,
        use_group_norm: bool = False,
        position_emb: bool = False,  # False -> Alibi
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)

        if position_emb:
            self.self_attn = MultiHeadAttention(
                dim=input_dim, num_heads=num_attention_heads, dropout=dropout
            )
        else:
            self.self_attn = MultiHeadAttentionAlibi(
                dim=input_dim, num_heads=num_attention_heads, dropout=dropout
            )
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=False, # True -> False
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        x = input
        residual = input
        x = self.self_attn_layer_norm(x)
        x = self.ffn1(x)
        x = x * 0.5 + residual


        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x, _ = self.self_attn(
            Q=x,
            K=x,
            V=x,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn2(x)
        x = x * 0.5 + residual
        return x

class Conformer(nn.Module):
    """Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_size (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_size=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_size)  # (batch, num_frames, input_size)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        use_pos_emb: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_size,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                    position_emb=use_pos_emb
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_size)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_size)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        # encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            # x = layer(x, encoder_padding_mask)
            x = layer(x)
        return x.transpose(0, 1), lengths



def _test_conformer():
    model = Conformer(
        input_size=256, 
        num_heads=8, 
        ffn_dim=256*3,
        num_layers=4,
        depthwise_conv_kernel_size=3
        )
    lengths = torch.tensor([250] * 4)
    x = torch.rand((4, 250, 256))
    with torch.no_grad():
        z, lengths = model(x, lengths)
    print("z: ", tuple(z.shape))
    print("lengths: ", tuple(lengths.shape))
    pprint(model)


if __name__ == "__main__":

    _test_conformer()
