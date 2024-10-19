# implement a vision transformer model

import torch
import torch.nn as nn
from torchvision.models import vision_transformer
from typing import *
from functools import partial
import math

# 1. MLP
class MLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.GELU,
        dropout: float = 0.0,
    ):
        """
        Parameters:
        - in_channels: int, number of input channels
        - hidden_channels: List[int], list of hidden layer sizes
        - out_channels: int, number of output channels
        - dropout: float, dropout rate
        """
        super().__init__()
        self.layers = nn.Sequential()
        if len(hidden_channels) == 0:
            self.layers.add_module("linear", nn.Linear(in_channels, out_channels))
            self.layers.add_module("dropout", nn.Dropout(dropout))
        else:
            self.layers.add_module("linear_1", nn.Linear(in_channels, hidden_channels[0]))
            self.layers.add_module("activation_1", activation_layer())
            self.layers.add_module("dropout_1", nn.Dropout(dropout))
            for i in range(1, len(hidden_channels) - 1):
                self.layers.add_module(f"linear_{i+1}", nn.Linear(hidden_channels[i-1], hidden_channels[i]))
                self.layers.add_module(f"activation_{i+1}", activation_layer())
                self.layers.add_module(f"dropout_{i+1}", nn.Dropout(dropout))
            self.layers.add_module("linear_out", nn.Linear(hidden_channels[-1], out_channels))
            self.layers.add_module("dropout_out", nn.Dropout(dropout))

        # initialize weights using xavier_uniform_ and biases using normal_
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    

# 2. EncoderLayer
class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Parameters:
        - num_heads: int, number of attention heads
        - hidden_dim: int, token feature dimension in the transformer
        - mlp_dim: int, hidden dimension in the MLP
        - dropout: float, dropout rate
        - attention_dropout: float, dropout rate in the attention layer
        """
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = norm_layer(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(hidden_dim, [mlp_dim], hidden_dim, dropout=dropout)
        self.norm2 = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm1(input)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = self.dropout1(x)
        x = x + input

        y = self.norm2(x)
        y = self.mlp(y)
        y = y + x
        return y


# 3. Encoder
class Encoder(torch.nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Parameters:
        - seq_length: int, length of the input sequence
        - num_layers: int, number of layers in the encoder
        - num_heads: int, number of attention heads
        - hidden_dim: int, token feature dimension in the transformer
        - mlp_dim: int, hidden dimension in the MLP
        - dropout: float, dropout rate
        - attention_dropout: float, dropout rate in the attention layer
        """
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderLayer(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


# 4. VisionTransformer
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Parameters:
        - image_size: int, size of the input image, assuming the image is square
        - patch_size: int, size of the image patch, image_size should be divisible by patch_size
        - num_layers: int, number of layers in the encoder
        - num_heads: int, number of attention heads
        - hidden_dim: int, token feature dimension in the transformer
        - mlp_dim: int, hidden dimension in the MLP
        - dropout: float, dropout rate
        - attention_dropout: float, dropout rate in the attention layer
        - num_classes: int, number of classes in the classification task

        This module first convolutionally embeds the input image into a sequence
        of token features, then applies the transformer encoder to the token 
        sequence, in the end, applies a linear layer to get the classification 
        result.
        """

        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.seq_length = (image_size // patch_size) ** 2 + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.encoder = Encoder(
            self.seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

        fan_in = self.conv.in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]
        nn.init.trunc_normal_(self.conv.weight, std=math.sqrt(1 / fan_in))
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(c == 3, "Input tensor does not have 3 channels as expected!")
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.permute(0, 2, 1)

        # (n, (n_h * n_w), hidden_dim) -> (n, seq_length, hidden_dim)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # (n, seq_length, hidden_dim) -> (n, seq_length, hidden_dim)
        x = self.encoder(x)

        # (n, seq_length, hidden_dim) -> (n, hidden_dim)
        x = x[:, 0]

        # (n, hidden_dim) -> (n, num_classes)
        x = self.head(x)

        return x

