import torch
from torch.optim import Adam
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from model import HAGCM
import torch.nn as nn
import numpy as np
import os

# Giải quyết cảnh báo liên quan đến OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ====================
# Định nghĩa siêu tham số
# ====================
BERT_MODEL_NAME = 'bert-base-uncased'  # Tên mô hình BERT được sử dụng
HIDDEN_DIM = 768  # Số chiều ẩn của mạng
GNN_OUTPUT_DIM = 768  # Kích thước đầu ra của GNN
BATCH_SIZE = 3  # Kích thước batch
LEARNING_RATE = 2e-5  # Hệ số học
NUM_EPOCHS = 5  # Số lượng epochs huấn luyện
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sử dụng GPU nếu có

# Tải dữ liệu đã tiền xử lý
texts, attention_masks, labels_1, labels_2 = torch.load('processed_data.pt')

# Khởi tạo mô hình
model = HAGCM(bert_model_name=BERT_MODEL_NAME, hidden_dim=HIDDEN_DIM, gnn_output_dim=GNN_OUTPUT_DIM)
model = model.to(DEVICE)

# Tạo đồ thị nhãn từ dữ liệu `label1` và `label2`
unique_labels = list(set(labels_1.tolist() + labels_2.tolist()))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Tạo danh sách các cạnh cho đồ thị nhãn
edges = []
for label1, label2 in zip(labels_1, labels_2):
    parent_idx = label_to_index[label1.item()]
    child_idx = label_to_index[label2.item()]
    edges.append([parent_idx, child_idx])

# Chuyển các cạnh sang định dạng tensor
edges = torch.tensor(edges, dtype=torch.long).T

# Khởi tạo embedding ngẫu nhiên cho nhãn
label_embeddings = torch.rand((len(unique_labels), 768))  # Kích thước ban đầu của nhãn

# Tạo đối tượng đồ thị cho nhãn
graph_data = Data(x=label_embeddings, edge_index=edges)

# Sử dụng MultiLabelBinarizer để tạo nhãn nhị phân
label_list = []
for label1, label2 in zip(labels_1, labels_2):
    labels = [label1.item(), label2.item()]
    label_list.append(labels)

mlb = MultiLabelBinarizer(classes=unique_labels)
labels = mlb.fit_transform(label_list)
labels = torch.tensor(labels, dtype=torch.float32)

# Chia dữ liệu thành tập train và validation theo tỷ lệ 8:2
train_texts, val_texts, train_masks, val_masks, train_labels, val_labels = train_test_split(
    texts, attention_masks, labels, test_size=0.2, random_state=42
)

# Tạo TensorDataset cho train và validation
train_dataset = TensorDataset(train_texts, train_masks, train_labels)
val_dataset = TensorDataset(val_texts, val_masks, val_labels)

# Tạo DataLoader cho train và validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Trình tối ưu hóa và hàm loss
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# Vòng lặp huấn luyện
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0  # Biến để lưu tổng loss cho epoch
    all_train_labels = []
    all_train_preds = []

    # Duyệt qua từng batch trong train_loader
    for i, batch in enumerate(train_loader):
        batch_texts, batch_masks, batch_labels = [b.to(DEVICE) for b in batch]

        # Forward pass
        output = model(input_ids=batch_texts, attention_mask=batch_masks, edge_index=graph_data.edge_index.to(DEVICE), label_embeddings=graph_data.x.to(DEVICE))

        # Tính toán loss
        loss = criterion(output, batch_labels)

        # Backward và cập nhật tham số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Cộng dồn loss cho mỗi batch vào tổng loss
        running_loss += loss.item()

        # In ra loss sau mỗi batch
        print(f"Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()}")

        # Lưu lại các giá trị dự đoán và nhãn thực tế cho F1 score sau epoch
        all_train_labels.append(batch_labels.detach().cpu().numpy())
        all_train_preds.append(torch.sigmoid(output).detach().cpu().numpy())

    # Sau mỗi epoch, tính toán F1-micro và F1-macro cho tập train
    model.eval()
    all_train_labels = np.vstack(all_train_labels)  # Chuyển danh sách thành ma trận
    all_train_preds = np.vstack(all_train_preds)  # Chuyển danh sách thành ma trận
    predicted_train_labels = (all_train_preds > 0.5).astype(int)

    # Tính toán F1-micro và F1-macro cho train
    f1_micro_train = f1_score(all_train_labels, predicted_train_labels, average='micro')
    f1_macro_train = f1_score(all_train_labels, predicted_train_labels, average='macro')

    # Đánh giá trên tập validation sau mỗi epoch
    val_loss = 0.0
    all_val_labels = []
    all_val_preds = []

    with torch.no_grad():
        for val_batch in val_loader:
            val_texts, val_masks, val_labels = [b.to(DEVICE) for b in val_batch]

            # Forward pass trên tập validation
            val_output = model(input_ids=val_texts, attention_mask=val_masks, edge_index=graph_data.edge_index.to(DEVICE), label_embeddings=graph_data.x.to(DEVICE))

            # Tính toán loss cho tập validation
            batch_val_loss = criterion(val_output, val_labels)
            val_loss += batch_val_loss.item()

            # Lưu lại các giá trị dự đoán và nhãn thực tế cho F1 score
            all_val_labels.append(val_labels.detach().cpu().numpy())
            all_val_preds.append(torch.sigmoid(val_output).detach().cpu().numpy())

    # Tính toán F1-micro và F1-macro cho tập validation
    all_val_labels = np.vstack(all_val_labels)  # Chuyển danh sách thành ma trận
    all_val_preds = np.vstack(all_val_preds)  # Chuyển danh sách thành ma trận
    predicted_val_labels = (all_val_preds > 0.5).astype(int)

    f1_micro_val = f1_score(all_val_labels, predicted_val_labels, average='micro')
    f1_macro_val = f1_score(all_val_labels, predicted_val_labels, average='macro')

    # In ra loss và F1 scores cho mỗi epoch
    epoch_loss = running_loss / len(train_loader)  # Trung bình loss cho cả epoch
    val_loss = val_loss / len(val_loader)  # Trung bình loss cho tập validation

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {epoch_loss}, Train F1_micro: {f1_micro_train}, Train F1_macro: {f1_macro_train}")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Val Loss: {val_loss}, Val F1_micro: {f1_micro_val}, Val F1_macro: {f1_macro_val}")

print("Huấn luyện hoàn tất!")
