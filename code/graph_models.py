import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GATConv

class HierarchicalAttention(nn.Module):
    """계층적 관계에 대한 어텐션 메커니즘"""
    def __init__(self, hidden_dim):
        super(HierarchicalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, parent_embeds, child_embeds):
        """
        부모-자식 노드 간 어텐션 계산
        
        Args:
            parent_embeds: 부모 노드 임베딩 [n_parents, hidden_dim]
            child_embeds: 자식 노드 임베딩 [n_children, hidden_dim]
            
        Returns:
            weighted_embeds: 어텐션 가중치가 적용된 임베딩
        """
        n_parents = parent_embeds.size(0)
        n_children = child_embeds.size(0)
        
        # 각 부모-자식 쌍에 대한 어텐션 점수 계산
        parent_expanded = parent_embeds.unsqueeze(1).expand(n_parents, n_children, -1)
        child_expanded = child_embeds.unsqueeze(0).expand(n_parents, n_children, -1)
        
        # 연결 및 어텐션 점수 계산
        concat = torch.cat([parent_expanded, child_expanded], dim=2)
        attn_scores = self.attn(concat.view(-1, concat.size(2))).view(n_parents, n_children)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 가중 합 계산
        weighted_embeds = torch.bmm(attn_weights.unsqueeze(1), child_embeds.unsqueeze(0).expand(n_parents, n_children, -1))
        
        return weighted_embeds.squeeze(1)

class TemporalAttentionLayer(nn.Module):
    """시간적 어텐션 레이어"""
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(TemporalAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        """
        시간적 어텐션 적용
        
        Args:
            x: 입력 시퀀스 [batch_size, seq_len, input_dim]
            
        Returns:
            attn_output: 어텐션이 적용된 시퀀스
        """
        # 차원 변환: [batch_size, seq_len, input_dim] -> [seq_len, batch_size, hidden_dim]
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        
        # 멀티헤드 어텐션 적용
        attn_output, _ = self.attn(x, x, x)
        
        # 차원 복원: [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        attn_output = attn_output.permute(1, 0, 2)
        
        return attn_output

class SpatioTemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_layers=2):
        super(SpatioTemporalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        
        # 입력 변환 레이어
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        
        # RGCN 레이어 (계층적 관계 모델링)
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations)
            for _ in range(num_layers)
        ])
        
        # GAT 레이어 (노드 간 영향력 파악)
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 시간적 어텐션 레이어
        self.temporal_attn = TemporalAttentionLayer(hidden_dim, hidden_dim)
        
        # 계층적 어텐션 레이어
        self.hierarchical_attn = HierarchicalAttention(hidden_dim)
        
        # 출력 변환 레이어
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        batch_size, num_nodes, seq_len = x.size(0), x.size(1), x.size(2)
        
        # 입력 변환: [batch_size, num_nodes, seq_len] -> [batch_size, num_nodes, hidden_dim]
        x = self.input_fc(x.view(batch_size * num_nodes, seq_len)).view(batch_size, num_nodes, self.hidden_dim)
        
        # 시간적 어텐션 적용
        x = self.temporal_attn(x)
        
        # 그래프 컨볼루션 적용 (각 시간 단계에 대해)
        for i in range(self.num_layers):
            # RGCN: 계층적 관계 모델링
            x_rgcn = x.view(-1, self.hidden_dim)
            x_rgcn = self.rgcn_layers[i](x_rgcn, edge_index, edge_type)
            x_rgcn = x_rgcn.view(batch_size, num_nodes, self.hidden_dim)
            
            # GAT: 노드 간 영향력 파악
            x_gat = x.view(-1, self.hidden_dim)
            x_gat = self.gat_layers[i](x_gat, edge_index)
            x_gat = x_gat.view(batch_size, num_nodes, self.hidden_dim)
            
            # 결합 및 비선형성 적용
            x = F.relu(x_rgcn + x_gat)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # 출력 변환: [batch_size, num_nodes, hidden_dim] -> [batch_size, num_nodes, output_dim]
        x = self.output_fc(x.view(batch_size * num_nodes, self.hidden_dim)).view(batch_size, num_nodes, self.output_dim)
        
        return x

# 모델 초기화
input_dim = 28  # 28일 데이터 (조정 필요)
hidden_dim = 64
output_dim = 28  # 28일 예측
num_relations = 4  # geo_hierarchy, prod_hierarchy, agg_hierarchy, cross_hierarchy

model = SpatioTemporalGNN(input_dim, hidden_dim, output_dim, num_relations)