import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MessagePassing(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=64, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Edge type specific transformations
        self.edge_type_embeddings = nn.Embedding(3, hidden_dim)  # 3 types: up, down, cross
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, node_features, edges, edge_types, num_nodes=None):
        # 입력 데이터를 텐서로 변환
        device = self.query.weight.device
        
        if isinstance(node_features, np.ndarray):
            node_features = torch.FloatTensor(node_features).to(device)
        
        if isinstance(edges, np.ndarray):
            edges = torch.LongTensor(edges).to(device)
            
        if isinstance(edge_types, np.ndarray):
            edge_types = torch.LongTensor(edge_types).to(device)
            
        if num_nodes is None:
            num_nodes = node_features.shape[0]
        
        # Prepare attention inputs
        Q = self.query(node_features).view(-1, self.num_heads, self.head_dim)
        K = self.key(node_features).view(-1, self.num_heads, self.head_dim)
        V = self.value(node_features).view(-1, self.num_heads, self.head_dim)
        
        # Initialize output
        messages = torch.zeros((num_nodes, self.input_dim), device=device)
        
        # Process each edge type
        for edge_type in range(3):
            mask = edge_types == edge_type
            if not mask.any():
                continue
                
            type_edges = edges[mask]
            edge_embedding = self.edge_type_embeddings(torch.tensor(edge_type, device=device))
            
            # Source and target node features
            source_nodes = type_edges[:, 0]
            target_nodes = type_edges[:, 1]
            
            # Compute attention scores
            scores = torch.matmul(Q[source_nodes], K[target_nodes].transpose(-2, -1))
            scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float, device=device))
            attention = F.softmax(scores, dim=-1)
            
            # Apply attention
            msg = torch.matmul(attention, V[target_nodes])
            msg = msg.reshape(-1, self.hidden_dim)
            
            # Add edge type information
            msg = msg + edge_embedding
            
            # Aggregate messages
            for i, src_idx in enumerate(source_nodes):
                messages[src_idx] += self.output_projection(msg[i])
        
        return messages

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.layers = nn.ModuleList([
            MessagePassing(
                hidden_dim if i > 0 else input_dim,
                hidden_dim,
                num_heads=num_heads
            ) for i in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, edges, edge_types):
        # 데이터 자동 변환
        device = self.input_projection.weight.device
        
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
            
        # Store original input for residual connection
        original_x = x
        
        # Initial projection
        hidden = self.input_projection(x)
        
        # Message passing layers
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            # Apply message passing
            message = layer(hidden, edges, edge_types, x.size(0))
            # Residual connection and normalization
            hidden = layer_norm(hidden + message)
            hidden = F.relu(hidden)
        
        # Final projection and residual connection
        adjustment = self.output_projection(hidden)
        return original_x + adjustment