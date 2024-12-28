import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import degree
from torch.nn.functional import dropout

class Model_GCN(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size, num_heads=8, dropout=0.2, device="cpu"):
        super(Model_GCN_v2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Initialize the GCN layers
        for i in range(num_layers):
            self.gcn_layers.append(
                GCNConv(
                    in_channels=self.hidden_size if i > 0 else input_size,
                    out_channels=self.hidden_size,
                    bias=False
                )
            )
            self.dropouts.append(nn.Dropout(dropout))
        
        # Initialize the GAT layers
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    heads=num_heads,
                    dropout=dropout
                )
            )
            self.dropouts.append(nn.Dropout(dropout))
        
        # Initialize the final regression layer
        self.final_linear = nn.Linear(self.hidden_size, output_size)
        
        # Initialize the degree scaling factor for GCN layers
        self.degree_scalers = nn.Parameter(
            torch.ones(num_layers) * torch.sqrt(torch.tensor(degree(torch.eye(hidden_size).to(device), dtype=torch.float32))
        ))
    
    def forward(self, data):
        # Get the adjacency matrix and features from the data
        x, edge_index = data.x, data.edge_index
        
        # Normalize adjacency matrix for GCN layers
        adj_norm = self.degree_scalers[0].view(-1, 1) * adj_t_norm = degree(edge_index, x.size(0), norm="row")
        adj_t_norm = adj_norm + torch.eye(x.size(0)).to(self.device)
        
        # Apply GCN layers
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, adj_t_norm)
            x = self.dropouts[i](x)
            x = x * adj_t_norm.view(-1, 1)
            x = self.ELU(x)
        
        # Apply GAT layers
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            x = self.dropouts[i + self.num_layers](x)
            x = self.ELU(x)
        
        # Global mean pooling for graph representation
        out = torch.mean(x, dim=0)
        
        # Apply final regression layer
        out = self.final_linear(out)
        
        return out