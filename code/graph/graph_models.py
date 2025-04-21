import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv, ASTGCN
   
class SpatioTemporalGNN(torch.nn.Module):
    def __init__(self, model_type, num_nodes=None, num_node_features=None, num_hidden_features=None, num_output_features=None, kernel_size=None, K=None):
        super(SpatioTemporalGNN, self).__init__()
            
        if model_type == 'STGCN':
            self.stgnn = STConv(
                num_nodes=num_nodes,
                in_channels=num_node_features,
                hidden_channels=num_hidden_features,
                out_channels=num_output_features,
                kernel_size=kernel_size,
                K=K,
            )
        # self.linear = torch.nn.Linear(num_output_features, 28) # single-shot prediction
        
    def forward(self, x, edge_index, edge_weight=None):
        h = self.stgnn(x, edge_index, edge_weight)
        # h = F.relu(h)
        # h = self.linear(h)
        return h