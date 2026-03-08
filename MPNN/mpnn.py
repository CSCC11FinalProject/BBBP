import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

from torch_geometric.nn import GINConv, global_add_pool # type: ignore
from torch_geometric.data import Data # type: ignore

class MPNN(nn.Module):
    def __init__(self, in_channels: int=20, feature_dim: int=7, hidden_channels: int=128, num_layers: int=4):
        assert num_layers > 1, "Model must have at least 2 layers."
        super(MPNN, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        def GINMPL(in_dim: int, out_dim: int) -> GINConv:
            return GINConv(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim), # layer norm is preferred over batch norm for GNNs
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            ))

        self.conv_blocks = nn.ModuleList(
            [GINMPL(in_channels, hidden_channels)] +
            [GINMPL(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )
        self.fusion_block = nn.Sequential(
            nn.Linear(hidden_channels + feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1), # single output for binary classification
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers-1)]) # of these
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch, u = data.x, data.edge_index, data.batch, data.u
        # first layer: conv + layernorm + relu
        h = self.conv_blocks[0](x, edge_index)
        h = F.relu(self.layer_norms[0](h))
        # middle layers: conv + layernorm + residual + relu + dropout
        for i in range(1, self.num_layers - 1):
            prev_h = h
            h = self.conv_blocks[i](h, edge_index)
            h = self.layer_norms[i](h)
            h = F.relu(h + prev_h)
            h = self.dropout(h)
        # last layer: conv + residual + relu (no layernorm, no dropout)
        prev_h = h
        h = self.conv_blocks[-1](h, edge_index)
        h = F.relu(h + prev_h)
        # global pooling
        graph_embeddings = global_add_pool(h, batch)
        combined_embeddings = torch.cat([graph_embeddings, u], dim=1)
        return self.fusion_block(combined_embeddings)  # logits for BCEWithLogitsLoss