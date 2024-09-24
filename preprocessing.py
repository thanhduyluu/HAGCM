import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, RobertaTokenizer

# Sử dụng tokenizer của BERT để mã hóa văn bản
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained("vinai/phobert-base")


def preprocess_data(input_file):
    # Đọc dữ liệu từ tệp CSV
    data = pd.read_csv(input_file, sep="\t")

    # Xử lý các nhãn (label1, label2)
    label_encoder_1 = LabelEncoder()
    label_encoder_2 = LabelEncoder()
    print(data)
    data['label1_encoded'] = label_encoder_1.fit_transform(data['lv1'])
    data['label2_encoded'] = label_encoder_2.fit_transform(data['lv2'])

    def tokenize_function(text):
        return tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=128)

    # Mã hóa văn bản
    data['text_encoded'] = data['sentences'].apply(lambda x: tokenize_function(x))

    # Chuyển dữ liệu thành dạng Tensor
    texts = torch.cat([d['input_ids'] for d in data['text_encoded']], dim=0)
    attention_masks = torch.cat([d['attention_mask'] for d in data['text_encoded']], dim=0)

    # Tạo tensor cho nhãn
    labels_1 = torch.tensor(data['label1_encoded'].values)
    labels_2 = torch.tensor(data['label2_encoded'].values)

    return texts, attention_masks, labels_1, labels_2


if __name__ == "__main__":
    input_file = 'test.csv'  # Thay thế bằng đường dẫn dữ liệu của bạn
    texts, attention_masks, labels_1, labels_2 = preprocess_data(input_file)
    torch.save((texts, attention_masks, labels_1, labels_2), 'processed_data.pt')
    print("Dữ liệu đã được tiền xử lý và lưu vào processed_data.pt")
