import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import *
from torch_geometric_temporal.nn.attention import *
   
class SpatioTemporalGNN(torch.nn.Module):
    def __init__(self, model_type, num_nodes=None, num_node_features=None, num_output_features=None, K=2):
        super(SpatioTemporalGNN, self).__init__()

        if model_type == 'A3TGCN':
            '''
            in_channels: number of node features
            out_channels: number of output features
            periods: number of time periods
            '''
            self.stgnn = A3TGCN(in_channels=num_node_features, out_channels=num_output_features, periods=28)
            
        elif model_type in ['GConvGRU', 'GConvLSTM', 'GCLSTM', 'DCRNN']:
            '''
            in_channels: number of node features
            out_channels: number of output features
            K: conv filter size
            '''
            if model_type == 'GConvGRU':
                self.stgnn = GConvGRU(in_channels=num_node_features, out_channels=num_output_features, K=K)
            elif model_type == 'GConvLSTM':
                self.stgnn = GConvLSTM(in_channels=num_node_features, out_channels=num_output_features, K=K)
            elif model_type == 'GCLSTM':
                self.stgnn = GCLSTM(in_channels=num_node_features, out_channels=num_output_features, K=K)
            elif model_type == 'DCRNN':
                self.stgnn = DCRNN(in_channels=num_node_features, out_channels=num_output_features, K=K)
    
        elif model_type == 'DyGrEncoder':
            '''
            conv_out_channels: number of conv channels
            conv_num_layers: number of convs
            conv_aggr: aggregation type
            lstm_out_channels: number of lstm channels
            lstm_num_layers: number of lstm neurons
            '''
            self.stgnn = DyGrEncoder(
                conv_out_channels=num_output_features,
                conv_num_layers=num_node_features,
                conv_aggr='add',
                lstm_out_channels=num_output_features,
                lstm_num_layers=num_node_features
            )
            
        elif model_type == 'EvolveGCNH':
            '''
            num_of_nodes: number of nodes
            in_channels: number of conv channels
            '''
            self.stgnn = EvolveGCNH(num_of_nodes=num_nodes, in_channels=num_output_features)
    
        elif model_type == 'EvolveGCNO':
            '''
            in_channels: number of conv channels
            '''
            self.stgnn = EvolveGCNO(in_channels=num_output_features)
    
        elif model_type == 'MPNNLSTM': 
            '''
            in_channels: number of node features
            hidden_size: number of output features
            num_nodes: number of nodes
            window: number of past time periods
            dropout: dropout rate
            '''
            self.stgnn = MPNNLSTM(
                in_channels=num_node_features, 
                hidden_size=num_output_features, 
                num_nodes=num_nodes,
                window=28,
                dropout=0.5
            )
            
        elif model_type == 'AGCRN': 
            '''
            number_of_nodes: number of nodes
            in_channels: number of node features
            out_channels: number of output features
            K: conv filter size
            embedding_dimensions: number of node features
            '''
            self.stgnn = AGCRN(
                number_of_nodes=num_nodes,
                in_channels=num_node_features,
                out_channels=num_output_features,
                K=K,
                embedding_dimensions=num_output_features
            )
            
        self.linear = torch.nn.Linear(num_output_features, 28) # single-shot prediction
        
    def forward(self, x, edge_index, edge_weight=None):
        h = self.stgnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h