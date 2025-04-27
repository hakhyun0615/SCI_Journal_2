import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(TemporalConv, self).__init__()
        r"""
        in_channels: feature_dim
        out_channels: num_channels
        kernel
            W: 1 (every node uses seperate filters independently)
            H: kernel_size
        """
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size)) # main information
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size)) # gate

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        input 
            X: (batch_size, time_steps, num_nodes, feature_dim)
        output 
            H: (batch_size, time_steps-kernel_size+1, num_nodes, num_channels)
        """
        X = X.permute(0, 3, 2, 1) # batch_size, feature_dim (C), num_nodes (W), time_steps (H))
        P = self.conv_1(X) # (batch_size, num_channels, num_nodes, time_steps-kernel_size+1)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = torch.mul(P, Q) # GLU (element-wise multiplication of P, Q): selective information flow
        return PQ

class STConv(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        num_temporal1_channels: int,
        num_spatial_channels: int,
        num_temporal2_channels: int,
        kernel_size: int,
        K: int
    ):
        super(STConv, self).__init__()
        r"""
        TemporalConv
            input: (batch_size, time_steps, num_nodes, feature_dim)
            output: (batch_size, time_steps-kernel_size+1, num_nodes, num_filters)
        ChebConv: 
            input: (batch_size, num_nodes, feature_dim)
            output: (batch_size, num_nodes, num_channels)
        """
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_temporal1_channels = num_temporal1_channels
        self.num_spatial_channels = num_spatial_channels
        self.kernel_size = kernel_size
        self.K = K

        self._temporal_conv1 = TemporalConv(
            in_channels=feature_dim,
            out_channels=num_temporal1_channels,
            kernel_size=kernel_size,
        )
        self._graph_conv = ChebConv(
            in_channels=num_temporal1_channels,
            out_channels=num_spatial_channels,
            K=K,
        )
        self._temporal_conv2 = TemporalConv(
            in_channels=num_spatial_channels,
            out_channels=num_temporal2_channels,
            kernel_size=kernel_size,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor) -> torch.FloatTensor:
        r"""
        intput 
            X: (batch_size, time_steps, num_nodes, feature_dim)
            edge_index: (2, num_edges)
        output
            T: (batch_size, time_steps-2*(kernel_size-1), num_nodes, num_temporal2_channels)
        """
        T_0 = self._temporal_conv1(X) # (batch_size, time_steps-kernel_size+1, num_nodes, num_temporal1_channels)
        T = torch.zeros(
            T_0.size(0),
            T_0.size(1),
            T_0.size(2),
            self.num_spatial_channels,
        ).to(T_0.device)
        for b in range(T_0.size(0)): # batch_size
            for t in range(T_0.size(1)): # time_steps-kernel_size+1
                T[b][t] = self._graph_conv(T_0[b][t], edge_index) # (num_nodes, num_spatial_channels)

        T = F.relu(T) # (batch_size, time_steps-kernel_size+1, num_nodes, num_spatial_channels)
        T = self._temporal_conv2(T) # (batch_size, time_steps-2*(kernel_size+1), num_nodes, num_temporal2_channels)
        T = T.permute(0, 2, 1, 3) # (batch_size (B), num_nodes (C), time_steps-2*(kernel_size+1) (H), num_temporal2_channels (W))
        T = self._batch_norm(T) # normalize by C
        T = T.permute(0, 2, 1, 3) # (batch_size, time_steps-2*(kernel_size+1), num_nodes, num_temporal2_channels)
        return T
    
class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        num_temporal1_channels: int = 64,
        num_spatial_channels: int = 16,
        num_temporal2_channels: int = 64,
        kernel_size: int = 3,
        K: int = 3,
    ):
        super(STGCN, self).__init__()
        
        self.stconv_blocks = nn.ModuleList()
        self.stconv_blocks.append(
            # first STConv block
            STConv(
                num_nodes=num_nodes,
                feature_dim=feature_dim,
                num_temporal1_channels=num_temporal1_channels,
                num_spatial_channels=num_spatial_channels,
                num_temporal2_channels=num_temporal2_channels,
                kernel_size=kernel_size,
                K=K,
            ),
            # second STConv block
            STConv(
                num_nodes=num_nodes,
                feature_dim=num_temporal2_channels,
                num_temporal1_channels=num_temporal1_channels,
                num_spatial_channels=num_spatial_channels,
                num_temporal2_channels=num_temporal2_channels,
                kernel_size=kernel_size,
                K=K,
            )
        )

        # align time dimension
        self.fc1= nn.Linear(
            in_features=28-2*(kernel_size+1),
            out_features=28
        )
        # align feature dimension
        self.fc2 = nn.Linear(
            in_features=num_temporal2_channels,  
            out_features=1 
        )
        self.relu = nn.ReLU()

    def forward(self, X, edge_index):
        """
        input
            X: (batch_size, time_steps, num_nodes, feature_dim)
            edge_index: (2, num_edges)
        output
            X: (batch_size, num_nodes, 28)
        """
        time_steps = X.size(1)
        for stconv in self.stconv_blocks:
            X = stconv(X, edge_index) # (batch_size, time_steps-2*(kernel_size+1), num_nodes, num_temporal2_channels)
        X = X.permute(0, 2, 3, 1)  # (batch_size, num_nodes, num_temporal2_channels, time_steps-2*(kernel_size+1))    
        X = self.fc1(X)  # (batch, num_nodes, num_temporal2_channels, 28)
        X = self.relu(X)
        X = X.permute(0, 1, 3, 2) # (batch, num_nodes, 28, num_temporal2_channels)
        X = self.fc2(X)  # (batch, num_nodes, 28, 1)
        X = X.squeeze(-1)  # (batch, num_nodes, 28)
       
        return X
