
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE

# ---------------------- 数据增强函数 ----------------------
def augment_raman_shift(spectra, wavenumber_range=(100, 1799), shift_range=(-0.5, 0.5), num_points=1700):
    start_wavenumber, end_wavenumber = wavenumber_range
    original_wavenumbers = np.linspace(start_wavenumber, end_wavenumber, num_points)
    shift = np.random.uniform(shift_range[0], shift_range[1])
    shifted_wavenumbers = original_wavenumbers + shift

    augmented_spectra = []
    for spectrum in spectra:
        interpolator = interp1d(original_wavenumbers, spectrum, kind='linear', fill_value="extrapolate")
        augmented_spectrum = interpolator(shifted_wavenumbers)
        augmented_spectra.append(augmented_spectrum)

    return np.array(augmented_spectra)

def balance_dataset(spectra, labels, desc_list, wavenumber_range=(100, 1799), shift_range=(-0.5, 0.5)):
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    print(f"Max sample count: {max_count}")

    augmented_spectra = []
    augmented_labels = []
    augmented_desc = []

    for label in unique_labels:
        mask = labels == label
        class_spectra = spectra[mask]
        class_desc = np.array(desc_list)[mask]
        class_count = len(class_spectra)

        if class_count < max_count:
            num_to_augment = max_count - class_count
            indices = np.random.choice(class_count, num_to_augment, replace=True)
            spectra_to_augment = class_spectra[indices]
            desc_to_augment = class_desc[indices]
            augmented = augment_raman_shift(spectra_to_augment, wavenumber_range, shift_range)
            augmented_spectra.append(class_spectra)
            augmented_spectra.append(augmented)
            augmented_labels.extend([label] * class_count)
            augmented_labels.extend([label] * num_to_augment)
            augmented_desc.extend(class_desc.tolist())
            augmented_desc.extend(desc_to_augment.tolist())
        else:
            augmented_spectra.append(class_spectra)
            augmented_labels.extend([label] * class_count)
            augmented_desc.extend(class_desc.tolist())

    augmented_spectra = np.concatenate(augmented_spectra, axis=0)
    augmented_labels = np.array(augmented_labels)
    augmented_desc = augmented_desc
    return augmented_spectra, augmented_labels, augmented_desc

# ---------------------- Inception 模块定义 ----------------------
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, red_3x3, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, red_5x5, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ELU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.ELU()
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

# ---------------------- GoogLeNet 主干网络 ----------------------
class GoogLeNet(nn.Module):
    def __init__(self, input_size=1700, num_classes=-1):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.pre_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.inception1 = InceptionModule(64, 64, 96, 128, 16, 32, 32)    # output: 256
        self.inception2 = InceptionModule(256, 128, 128, 192, 32, 96, 64) # output: 480
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)  # output: 512
        self.inception4 = InceptionModule(512, 160, 112, 224, 24, 64, 64) # output: 512

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool(x)
        x = self.inception3(x)
        x = self.inception4(x)
        return x  # [B, 512, L]

# ---------------------- Cross-feature Fusion Module ----------------------
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = self.d_k ** 0.5

    def forward(self, query, key_value):
        B, T, D = query.shape
        L_r = key_value.shape[1]
        Q = self.query_proj(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(key_value).view(B, L_r, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(key_value).view(B, L_r, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V)
        attended = attended.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attended), attn

# ---------------------- Prediction ----------------------
class FCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------------- 整合所有模块 ----------------------
class MultimodalCrossAttentionClassifier(nn.Module):
    def __init__(self, num_classes, d_model=256, hidden_dim=128, dropout=0.5):
        super(MultimodalCrossAttentionClassifier, self).__init__()
        self.raman_extractor = GoogLeNet(num_classes=-1, input_size=1700)  # 输出 [B, 512, L]
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, d_model)  # 768 to 256
        self.raman_proj = nn.Linear(512, d_model)  # 512 to 256
        self.cross_attn1 = CrossAttention(d_model)
        self.cross_attn2 = CrossAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = FCClassifier(d_model * 2, hidden_dim, num_classes)

    def forward(self, spectra, input_ids, attention_mask, return_features=False):
        x_spec = spectra.unsqueeze(1)
        raman_out = self.raman_extractor(x_spec)
        raman_seq = raman_out.permute(0, 2, 1)
        raman_seq = self.raman_proj(raman_seq)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = text_outputs.last_hidden_state
        text_seq = self.text_proj(text_seq)

        fixed_len = 64
        B, L_r, D = raman_seq.shape
        B, L_t, D = text_seq.shape
        if L_r != fixed_len:
            raman_seq = F.adaptive_avg_pool1d(raman_seq.permute(0, 2, 1), fixed_len).permute(0, 2, 1)
        if L_t != fixed_len:
            text_seq = F.adaptive_avg_pool1d(text_seq.permute(0, 2, 1), fixed_len).permute(0, 2, 1)

        cross_out1, attn_weights1 = self.cross_attn1(raman_seq, text_seq)
        cross_out2, attn_weights2 = self.cross_attn2(text_seq, raman_seq)

        fused_features = torch.cat([cross_out1, cross_out2], dim=-1)
        fused_features = self.dropout(fused_features)

        logits = self.classifier(fused_features)

        if return_features:
            return logits, (attn_weights1, attn_weights2), fused_features
        return logits, (attn_weights1, attn_weights2)

# ---------------------- 参数初始化 ----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(m.bias)

# ---------------------- 数据加载与预处理 ----------------------
file_path = "../excellent_unoriented_data_top20_moveline.csv"
data = pd.read_csv(file_path)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["Name"].values)
spectra = data.iloc[:, 1:].values
names = data["Name"].values
scaler = StandardScaler()
spectra = scaler.fit_transform(spectra)
descriptions = {
    "Albite": "Raman spectrum peaks are located at '479','510',"
              "The peak at 479 is formed by tetrahedral ring vibrations,"
              "The peak at 510 is formed by Na+ vibrations.",
    "Almandine": "Raman spectrum peaks are located at '560','917',"
                 "The peak at 560 is formed by O-Si-O bending vibrations (δ),"
                 "The peak at 917 is formed by Si-O symmetric stretching vibrations (ν₁).",
    "Andradite": "Raman spectrum peaks are located at '514','873',"
                 "The peak at 514 is formed by O-Si-O bending vibrations (δ),"
                 "The peak at 873 is formed by Si-O asymmetric stretching vibrations (ν₃).",
    "Anorthite": "Raman spectrum peaks are located at '489','800',"
                 "The peak at 489 is formed by O-Si-O bending vibrations (δ),"
                 "The peak at 800 is formed by Si-O symmetric stretching vibrations (ν₁).",
    "Beryl": "Raman spectrum peaks are located at '687','1069',"
             "The peak at 687 is formed by Be-O vibrations,"
             "The peak at 1069 is formed by Si-O stretching vibrations.",
    "Calcite": "Raman spectrum peaks are located at '713','1084',"
               "The peak at 713 is formed by CO₃²⁻ in-plane bending vibrations (ν₄),"
               "The peak at 1084 is formed by CO₃²⁻ symmetric stretching vibrations (ν₁).",
    "Diamond": "Raman spectrum peak is located at '1332',"
               "The peak at 1332 is formed by C-C symmetric stretching vibrations (ν₁).",
    "Diopside": "Raman spectrum peaks are located at '671','1015',"
                "The peak at 671 is formed by Si-O-Si symmetric stretching vibrations (ν₁),"
                "The peak at 1015 is formed by Si-O₆ stretching vibrations.",
    "Dolomite": "Raman spectrum peaks are located at '725','1090',"
                "The peak at 725 is formed by CO₃²⁻ in-plane bending vibrations (ν₄),"
                "The peak at 1090 is formed by CO₃²⁻ symmetric stretching vibrations (ν₁).",
    "Elbaite": "Raman spectrum peaks are located at '711','1063',"
               "The peak at 711 is formed by BO₃²⁻ in-plane bending vibrations (ν₄),"
               "The peak at 1063 is formed by Si-O symmetric stretching vibrations (ν₁).",
    "Enstatite": "Raman spectrum peaks are located at '675','1037',"
                 "The peak at 675 is formed by Si-O-Si symmetric stretching vibrations (ν₁),"
                 "The peak at 1037 is formed by Si-O stretching vibrations.",
    "Epidote": "Raman spectrum peaks are located at '605','937',"
               "The peak at 605 is formed by Si-O symmetric stretching vibrations (ν₁),"
               "The peak at 937 is formed by Si-O symmetric stretching vibrations (ν₁).",
    "Fluorapatite": "Raman spectrum peaks are located at '581','965',"
                    "The peak at 581 is formed by PO₄³⁻ bending vibrations (ν₄),"
                    "The peak at 965 is formed by PO₄³⁻ symmetric stretching vibrations (ν₁).",
    "Forsterite": "Raman spectrum peaks are located at '827','965',"
                  "The peak at 827 is formed by SiO₄⁴⁻ symmetric stretching vibrations (ν₁),"
                  "The peak at 965 is formed by SiO₄⁴⁻ asymmetric stretching vibrations (ν₃).",
    "Grossular": "Raman spectrum peaks are located at '549','880',"
                 "The peak at 549 is formed by O-Si-O bending vibrations (ν₄),"
                 "The peak at 880 is formed by SiO₄⁴⁻ asymmetric stretching vibrations (ν₃).",
    "Marialite": "Raman spectrum peaks are located at '265','775',"
                 "The peak at 265 is formed by Na+ vibrations,"
                 "The peak at 775 is formed by SiO₄⁴⁻ symmetric stretching vibrations (ν₁).",
    "Muscovite": "Raman spectrum peak is located at '203',"
                 "The peak at 203 is formed by K+ vibrations.",
    "Pyrope": "Raman spectrum peaks are located at '552','912',"
              "The peak at 552 is formed by O-Si-O bending vibrations (ν₄),"
              "The peak at 912 is formed by Si-O asymmetric stretching vibrations (ν₃).",
    "Quartz": "Raman spectrum peaks are located at '206','465',"
              "The peak at 206 is formed by Si-O-Si asymmetric bending vibrations (ν₄),"
              "The peak at 465 is formed by Si-O symmetric stretching vibrations (ν₁).",
    "Wendwilsonite": "Raman spectrum peaks are located at '370','820',"
                     "The peak at 370 corresponds to H2O libration modes,"
                     "The peak at 820 is formed by the symmetric stretching vibration (ν₁) of [AsO4]3- groups."
}
desc_list = [descriptions.get(name, "No description available.") for name in names]

# 数据增强
spectra, labels, desc_list = balance_dataset(spectra, labels, desc_list)

# 数据划分
X_train, X_test, y_train, y_test, desc_train, desc_test = train_test_split(
    spectra, labels, desc_list, test_size=0.3, random_state=7941123, stratify=labels
)

# 检查类别分布
print(pd.Series(labels).value_counts())
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------------------- 加载文本模型与 Tokenizer ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=64)

# ---------------------- 定义模型 ----------------------
num_class = len(label_encoder.classes_)
dropout = 0.5
model = MultimodalCrossAttentionClassifier(
    num_classes=num_class,
    d_model=256,
    hidden_dim=128,
    dropout=dropout
).to(device)
model.apply(init_weights)

# ---------------------- 提取训练前特征 ----------------------
def extract_features(model, spectra, descriptions, tokenizer, device, batch_size=64, labels=None):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch_spectra = spectra[i:i + batch_size]
            batch_descriptions = descriptions[i:i + batch_size]
            batch_labels = labels[i:i + batch_size] if labels is not None else None
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device)
            inputs = tokenizer(batch_descriptions, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)

            _, _, fused_features = model(spectra_tensor, input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"], return_features=True)
            fused_features = fused_features.mean(dim=1)
            features_list.append(fused_features.cpu().numpy())
            if batch_labels is not None:
                labels_list.extend(batch_labels)

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list) if labels is not None else None

    expected_num_samples = len(spectra)
    if features.shape[0] != expected_num_samples:
        raise ValueError(f"Number of extracted features ({features.shape[0]}) does not match input samples ({expected_num_samples})")
    if labels is not None and len(labels) != expected_num_samples:
        raise ValueError(f"Number of labels ({len(labels)}) does not match input samples ({expected_num_samples})")

    return features, labels

batch_size = 64
# 提取训练前的测试集特征
pre_train_features, pre_train_labels = extract_features(model, X_test, desc_test, tokenizer, device, batch_size, y_test)
print(f"Pre-train features shape: {pre_train_features.shape}")
print(f"Pre-train labels shape: {pre_train_labels.shape}")
print(f"Unique pre-train labels: {np.unique(pre_train_labels, return_counts=True)}")

# 应用 UMAP（训练前）
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
pre_train_umap_features = umap_reducer.fit_transform(pre_train_features)
print(f"Pre-train UMAP features shape: {pre_train_umap_features.shape}")
unique_pre_train_umap_points = np.unique(pre_train_umap_features, axis=0)
print(f"Number of unique pre-train UMAP points: {unique_pre_train_umap_points.shape[0]}")

# ---------------------- 训练设置 ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
epochs = 100

def lr_lambda(current_epoch):
    warmup_epochs = 10
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / warmup_epochs
    else:
        return 0.5 * (1 + np.cos(np.pi * (current_epoch - warmup_epochs) / (epochs - warmup_epochs)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
batch_size = 128
max_grad_norm = 5.0

# 初始化 test_labels 和 test_preds 为全局变量
test_preds = []
test_labels = []

# ---------------------- 训练循环 ----------------------
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for i in range(0, len(X_train), batch_size):
        batch_spectra = X_train[i:i + batch_size]
        batch_labels = y_train_tensor[i:i + batch_size]
        batch_descriptions = desc_train[i:i + batch_size]
        spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device)
        inputs = tokenizer(batch_descriptions, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        optimizer.zero_grad()
        outputs, attn_weights = model(spectra_tensor, input_ids=inputs["input_ids"],
                                      attention_mask=inputs["attention_mask"])
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    train_acc = accuracy_score(all_labels, all_preds)
    train_losses.append(total_loss / len(X_train))
    train_accuracies.append(train_acc)

    # 清空 test_preds 和 test_labels 以存储当前 epoch 的测试结果
    test_preds = []
    test_labels = []
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_spectra = X_test[i:i + batch_size]
            batch_labels = y_test_tensor[i:i + batch_size]
            batch_descriptions = desc_test[i:i + batch_size]
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device)
            inputs = tokenizer(batch_descriptions, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            outputs, _ = model(spectra_tensor, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
    test_acc = accuracy_score(test_labels, test_preds)
    test_losses.append(test_loss / len(X_test))
    test_accuracies.append(test_acc)
    # scheduler.step()
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# ---------------------- 计算准确率、灵敏度、特异性 ----------------------
def compute_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = []
    specificity = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity.append(sens)

        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity.append(spec)

    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, avg_sensitivity, avg_specificity, sensitivity, specificity

# 计算测试集的指标
print(f"Length of test_labels: {len(test_labels)}")
print(f"Length of test_preds: {len(test_preds)}")
if len(test_labels) != len(test_preds):
    raise ValueError(f"test_labels (length {len(test_labels)}) and test_preds (length {len(test_preds)}) have different lengths!")
if len(test_labels) != len(X_test):
    raise ValueError(f"test_labels (length {len(test_labels)}) does not match the number of test samples ({len(X_test)})!")

test_accuracy, test_sensitivity, test_specificity, per_class_sensitivity, per_class_specificity = compute_metrics(
    test_labels, test_preds, num_class
)

# 输出结果
print("\nFinal Test Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Average Sensitivity (Recall): {test_sensitivity:.4f}")
print(f"Average Specificity: {test_specificity:.4f}")
print("\nPer-class Sensitivity (Recall):")
for i, sens in enumerate(per_class_sensitivity):
    print(f"Class {label_encoder.classes_[i]}: {sens:.4f}")
print("\nPer-class Specificity:")
for i, spec in enumerate(per_class_specificity):
    print(f"Class {label_encoder.classes_[i]}: {spec:.4f}")

# ---------------------- 提取训练后特征 ----------------------
post_train_features, post_train_labels = extract_features(model, X_test, desc_test, tokenizer, device, batch_size, y_test)
print(f"Post-train features shape: {post_train_features.shape}")
print(f"Post-train labels shape: {post_train_labels.shape}")
print(f"Unique post-train labels: {np.unique(post_train_labels, return_counts=True)}")

# 应用 UMAP（训练后）
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
post_train_umap_features = umap_reducer.fit_transform(post_train_features)
print(f"Post-train UMAP features shape: {post_train_umap_features.shape}")
unique_post_train_umap_points = np.unique(post_train_umap_features, axis=0)
print(f"Number of unique post-train UMAP points: {unique_post_train_umap_points.shape[0]}")

# ---------------------- 计算类内和类间距离以评估特征质量 ----------------------
def compute_class_distances(features, labels):
    num_classes = len(np.unique(labels))
    class_means = np.zeros((num_classes, features.shape[1]))
    intra_class_distances = []
    inter_class_distances = []

    for i in range(num_classes):
        class_features = features[labels == i]
        class_mean = np.mean(class_features, axis=0)
        class_means[i] = class_mean
        intra_dist = np.mean([np.linalg.norm(feat - class_mean) for feat in class_features])
        intra_class_distances.append(intra_dist)

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            inter_dist = np.linalg.norm(class_means[i] - class_means[j])
            inter_class_distances.append(inter_dist)

    avg_intra_class_dist = np.mean(intra_class_distances)
    avg_inter_class_dist = np.mean(inter_class_distances) if inter_class_distances else 0

    return avg_intra_class_dist, avg_inter_class_dist

pre_intra_dist, pre_inter_dist = compute_class_distances(pre_train_features, pre_train_labels)
post_intra_dist, post_inter_dist = compute_class_distances(post_train_features, post_train_labels)
print(f"Pre-train: Avg intra-class distance: {pre_intra_dist:.4f}, Avg inter-class distance: {pre_inter_dist:.4f}")
print(f"Post-train: Avg intra-class distance: {post_intra_dist:.4f}, Avg inter-class distance: {post_inter_dist:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 绘制训练前和训练后的 UMAP 可视化（带抖动，分开的两张图） ----------------------
# 添加抖动
jitter = 0.5
pre_train_umap_features_jittered = pre_train_umap_features + np.random.normal(0, jitter, pre_train_umap_features.shape)
post_train_umap_features_jittered = post_train_umap_features + np.random.normal(0, jitter, post_train_umap_features.shape)

# 绘制训练前的 UMAP 可视化
fig1, ax1 = plt.subplots(figsize=(12, 10))
scatter1 = ax1.scatter(pre_train_umap_features_jittered[:, 0], pre_train_umap_features_jittered[:, 1],
                       c=pre_train_labels, cmap='tab20', s=10, alpha=0.3)
ax1.set_title("UMAP Before Training (googlenet+bert)", fontsize=16)
ax1.set_xlabel("UMAP Component 1", fontsize=14)
ax1.set_ylabel("UMAP Component 2", fontsize=14)
ax1.set_xticks([])  # Hide x-axis ticks and labels
ax1.set_yticks([])  # Hide y-axis ticks and labels
# 添加颜色条
cbar1 = fig1.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar1.set_ticks(range(num_class))
cbar1.set_label('Classes', fontsize=14)
cbar1.set_ticklabels(label_encoder.classes_)
cbar1.ax.tick_params(rotation=45, labelsize=10)

plt.savefig("GoogLeNet_umap_before_training_jittered.png", bbox_inches='tight', dpi=1200)
plt.show()

# 绘制训练后的 UMAP 可视化
fig2, ax2 = plt.subplots(figsize=(12, 10))
scatter2 = ax2.scatter(post_train_umap_features_jittered[:, 0], post_train_umap_features_jittered[:, 1],
                       c=post_train_labels, cmap='tab20', s=10, alpha=0.3)
ax2.set_title("UMAP After Training (googlenet+bert)", fontsize=16)
ax2.set_xlabel("UMAP Component 1", fontsize=14)
ax2.set_ylabel("UMAP Component 2", fontsize=14)
ax2.set_xticks([])  # Hide x-axis ticks and labels
ax2.set_yticks([])  # Hide y-axis ticks and labels
# 添加颜色条
cbar2 = fig2.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar2.set_ticks(range(num_class))
cbar2.set_label('Classes', fontsize=14)
cbar2.set_ticklabels(label_encoder.classes_)
cbar2.ax.tick_params(rotation=45, labelsize=10)

plt.savefig("GoogLeNet_umap_after_training_jittered.png", bbox_inches='tight', dpi=1200)
plt.show()

# ---------------------- 子采样并绘制 UMAP 可视化 ----------------------
subsample_indices = []
samples_per_class = 5
for i in range(num_class):
    class_indices = np.where(post_train_labels == i)[0]
    subsample_indices.extend(class_indices[:samples_per_class])
subsample_indices = np.array(subsample_indices)
sub_pre_train_umap_features = pre_train_umap_features[subsample_indices]
sub_post_train_umap_features = post_train_umap_features[subsample_indices]
sub_pre_train_labels = pre_train_labels[subsample_indices]
sub_post_train_labels = post_train_labels[subsample_indices]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(bottom=0.25)

# 绘制训练前的 UMAP 可视化
fig1, ax1 = plt.subplots(figsize=(10, 8))
scatter1 = ax1.scatter(sub_pre_train_umap_features[:, 0], sub_pre_train_umap_features[:, 1],
                       c=sub_pre_train_labels, cmap='tab20', s=50, alpha=0.6)
ax1.set_title("UMAP Before Training (GoogLeNet+Bert)", fontsize=16)
ax1.set_xlabel("UMAP Component 1", fontsize=12)
ax1.set_ylabel("UMAP Component 2", fontsize=12)

# 添加颜色条
cbar1 = fig1.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar1.set_ticks(range(num_class))
cbar1.set_label('Classes', fontsize=14)
cbar1.set_ticklabels(label_encoder.classes_)
cbar1.ax.tick_params(rotation=45, labelsize=10)

plt.savefig("google_bert_umap_before_training.png", bbox_inches='tight', dpi=1200)
plt.show()

# 绘制训练后的 UMAP 可视化
fig2, ax2 = plt.subplots(figsize=(12, 10))
scatter2 = ax2.scatter(sub_post_train_umap_features[:, 0], sub_post_train_umap_features[:, 1],
                       c=sub_post_train_labels, cmap='tab20', s=50, alpha=0.6)
ax2.set_title("UMAP After Training (googlenet+bert)", fontsize=16)
ax2.set_xlabel("UMAP Component 1", fontsize=14)
ax2.set_ylabel("UMAP Component 2", fontsize=14)
ax2.set_xticks([])  # Hide x-axis ticks and labels
ax2.set_yticks([])  # Hide y-axis ticks and labels
# 添加颜色条
cbar2 = fig2.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar2.set_ticks(range(num_class))
cbar2.set_label('Classes', fontsize=14)
cbar2.set_ticklabels(label_encoder.classes_)
cbar2.ax.tick_params(rotation=45, labelsize=10)

plt.savefig("google_bert_umap_after_training.png", bbox_inches='tight', dpi=1200)
plt.show()

# 绘制混淆矩阵
class_names = label_encoder.classes_
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (GoogLeNet+Bert)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix_googlenet_Bert_UMP.png", bbox_inches='tight')
plt.show()