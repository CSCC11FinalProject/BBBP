import torch # type: ignore
import torch.nn as nn # type: ignore
from torch_geometric.nn import NNConv # type: ignore
from torch_geometric.data import Data # type: ignore
from torch_geometric.utils import to_dense_batch # type: ignore

class MPNN(nn.Module):
    def __init__(
        self, 
        atom_dim: int = 29, 
        bond_dim: int = 7, 
        tabular_dim: int = 7,
        message_units: int = 64, 
        message_steps: int = 4, 
        num_attention_heads: int = 8, 
        dense_units: int = 512
    ):
        super(MPNN, self).__init__()
        self.message_steps = message_steps
        self.message_units = message_units
        
        self.pad_length = max(0, message_units - atom_dim)
        self.node_dim = atom_dim + self.pad_length
        
        edge_nn = nn.Linear(bond_dim, self.node_dim * self.node_dim)
        self.conv = NNConv(self.node_dim, self.node_dim, edge_nn, aggr='add', root_weight=False)
        self.gru = nn.GRUCell(self.node_dim, self.node_dim)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.node_dim, 
            num_heads=num_attention_heads, 
            batch_first=True
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(self.node_dim, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, self.node_dim)
        )
        self.layernorm_1 = nn.LayerNorm(self.node_dim)
        self.layernorm_2 = nn.LayerNorm(self.node_dim)
        
        # FUSION: Transformer output (node_dim) + your tabular features (tabular_dim)
        self.classification = nn.Sequential(
            nn.Linear(self.node_dim + tabular_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(dense_units, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch, u = data.x, data.edge_index, data.edge_attr, data.batch, data.u
        
        if self.pad_length > 0:
            padding = torch.zeros(x.size(0), self.pad_length, device=x.device)
            x = torch.cat([x, padding], dim=-1)
            
        h = x
        for _ in range(self.message_steps):
            m = self.conv(h, edge_index, edge_attr)
            h = self.gru(m, h)
            
        dense_x, mask = to_dense_batch(h, batch)
        key_padding_mask = ~mask 
        
        attn_out, _ = self.self_attention(
            query=dense_x, 
            key=dense_x, 
            value=dense_x, 
            key_padding_mask=key_padding_mask
        )
        
        proj_input = self.layernorm_1(dense_x + attn_out)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        
        mask_expanded = mask.unsqueeze(-1).float()
        sum_pool = torch.sum(proj_output * mask_expanded, dim=1)
        valid_node_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = sum_pool / valid_node_counts
        
        # Concatenate Graph embedding with the 7 Tabular Features
        combined_embeddings = torch.cat([pooled, u], dim=1)
        
        return self.classification(combined_embeddings)