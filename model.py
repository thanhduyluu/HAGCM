import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GCNConv


class HAGCM(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=128, gnn_output_dim=64):
        super(HAGCM, self).__init__()
        # Sử dụng BERT để mã hóa văn bản
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_dim = self.bert.config.hidden_size

        # Lớp chú ý phân cấp (Hierarchical Attention)
        self.attention = nn.Sequential(
            nn.Linear(self.bert_hidden_dim, hidden_dim),
            nn.Tanh(),  # Áp dụng phi tuyến tính
            nn.Linear(hidden_dim, 1)  # Trả về trọng số chú ý cho mỗi token
        )

        # Lớp giảm chiều cho nhãn phân cấp (từ 768 xuống 128)
        self.reduce_dim = nn.Linear(self.bert_hidden_dim, hidden_dim)

        # Lớp giảm chiều biểu diễn văn bản sau khi áp dụng chú ý (từ 768 xuống 64)
        self.reduce_text_dim = nn.Linear(self.bert_hidden_dim, gnn_output_dim)

        # Mạng nơ-ron đồ thị (GNN) để mã hóa nhãn phân cấp
        self.gnn_conv1 = GCNConv(hidden_dim, gnn_output_dim)
        self.gnn_conv2 = GCNConv(gnn_output_dim, gnn_output_dim)

    def forward(self, input_ids, attention_mask, edge_index, label_embeddings):
        # Bước 1: Mã hóa văn bản bằng BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_x = bert_output.last_hidden_state  # Lấy toàn bộ biểu diễn [batch_size, seq_len, 768]

        # Bước 2: Tính toán trọng số chú ý cho từng token
        attn_scores = self.attention(h_x)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # Trọng số chú ý, [batch_size, seq_len, 1]

        # Áp dụng trọng số chú ý lên biểu diễn của các token
        h_x_weighted = torch.sum(attn_weights * h_x, dim=1)  # Biểu diễn có trọng số, [batch_size, 768]

        # Bước 3: Giảm chiều của nhãn phân cấp từ 768 xuống 128
        label_embeddings = self.reduce_dim(label_embeddings)  # [num_labels, 128]

        # Bước 4: Giảm chiều biểu diễn văn bản từ 768 xuống 64 để khớp với kích thước nhãn
        h_x_weighted = self.reduce_text_dim(h_x_weighted)  # [batch_size, 64]

        # Bước 5: GNN để mã hóa nhãn phân cấp
        x = F.relu(self.gnn_conv1(label_embeddings, edge_index))
        x = self.gnn_conv2(x, edge_index)

        # Bước 6: Tính toán độ tương đồng giữa văn bản và nhãn
        h_x_normalized = F.normalize(h_x_weighted, p=2, dim=-1)  # Chuẩn hóa biểu diễn văn bản
        x_normalized = F.normalize(x, p=2, dim=-1)  # Chuẩn hóa biểu diễn nhãn

        similarity = torch.matmul(h_x_normalized, x_normalized.T)  # [batch_size, num_labels]

        return similarity
