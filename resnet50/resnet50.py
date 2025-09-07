
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

# ---------------------- ResNet50 定义 ----------------------
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm1d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, blocks=[3, 4, 6, 3], num_classes=20, expansion=4):
        super(ResNet50, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=1, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * expansion, num_classes)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.avgpool(x)
        features = torch.flatten(features, 1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

# ---------------------- 参数初始化 ----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
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
desc_list = ["Placeholder description" for _ in names]

# 数据增强
spectra, labels, desc_list = balance_dataset(spectra, labels, desc_list)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    spectra, labels, test_size=0.3, random_state=42, stratify=labels
)

# 检查类别分布
print(pd.Series(labels).value_counts())
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------------------- 定义模型 ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
num_classes = len(label_encoder.classes_)
model = ResNet50(num_classes=num_classes).to(device)
model.apply(init_weights)

# ---------------------- 提取特征 ----------------------
def extract_features(model, spectra, device, batch_size=32):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch_spectra = spectra[i:i + batch_size]
            batch_labels = y_test[i:i + batch_size] if len(spectra) == len(y_test) else None
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device).unsqueeze(1)
            _, features = model(spectra_tensor, return_features=True)
            features_list.append(features.cpu().numpy())
            if batch_labels is not None:
                labels_list.extend(batch_labels)

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list) if labels_list else None

    return features, labels

# 提取训练前的测试集特征
pre_train_features, pre_train_labels = extract_features(model, X_test, device, batch_size=32)
print(f"Pre-train features shape: {pre_train_features.shape}")
print(f"Pre-train labels shape: {pre_train_labels.shape}")

# 应用 UMAP（训练前）
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
pre_train_umap_features = umap_reducer.fit_transform(pre_train_features)
print(f"Pre-train UMAP features shape: {pre_train_umap_features.shape}")

# ---------------------- 训练设置 ----------------------
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss()  # weight=class_weights_tensor can be added if needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 50
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
batch_size = 128
max_grad_norm = 2.0

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
        spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(spectra_tensor)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    train_acc = accuracy_score(all_labels, all_preds)
    train_losses.append(total_loss / (len(X_train) / batch_size))
    train_accuracies.append(train_acc)

    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_spectra = X_test[i:i + batch_size]
            batch_labels = y_test_tensor[i:i + batch_size]
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device).unsqueeze(1)
            outputs = model(spectra_tensor)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
    test_acc = accuracy_score(test_labels, test_preds)
    test_losses.append(test_loss / (len(X_test) / batch_size))
    test_accuracies.append(test_acc)
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# 提取训练后的测试集特征
post_train_features, post_train_labels = extract_features(model, X_test, device, batch_size=32)
print(f"Post-train features shape: {post_train_features.shape}")
print(f"Post-train labels shape: {post_train_labels.shape}")

# 应用 UMAP（训练后）
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
post_train_umap_features = umap_reducer.fit_transform(post_train_features)
print(f"Post-train UMAP features shape: {post_train_umap_features.shape}")

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
test_accuracy, test_sensitivity, test_specificity, per_class_sensitivity, per_class_specificity = compute_metrics(
    test_labels, test_preds, num_classes
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

# 绘制混淆矩阵
class_names = label_encoder.classes_
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (ResNet50)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix_resnet50_augmented.png", bbox_inches='tight')
plt.show()

# ---------------------- 绘制训练前和训练后的 UMAP 可视化（带抖动，分开的两张图） ----------------------
# 添加抖动
jitter = 0.5
pre_train_umap_features_jittered = pre_train_umap_features + np.random.normal(0, jitter, pre_train_umap_features.shape)
post_train_umap_features_jittered = post_train_umap_features + np.random.normal(0, jitter, post_train_umap_features.shape)

# 绘制训练前的 UMAP 可视化
fig1, ax1 = plt.subplots(figsize=(12, 10))
scatter1 = ax1.scatter(pre_train_umap_features_jittered[:, 0], pre_train_umap_features_jittered[:, 1],
                       c=pre_train_labels, cmap='tab20', s=10, alpha=0.3)
ax1.set_title("UMAP Before Training (ResNet50)", fontsize=16)
ax1.set_xlabel("UMAP Component 1", fontsize=14)
ax1.set_ylabel("UMAP Component 2", fontsize=14)
ax1.set_xticks([])  # Hide x-axis ticks and labels
ax1.set_yticks([])  # Hide y-axis ticks and labels
# 添加颜色条
cbar1 = fig1.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar1.set_ticks(range(num_classes))
cbar1.set_label('Classes', fontsize=14)
cbar1.set_ticklabels(label_encoder.classes_)
cbar1.ax.tick_params(rotation=45, labelsize=10)

plt.savefig("ResNet50_umap_before_training_jittered.png", bbox_inches='tight', dpi=1200)
plt.show()

# 绘制训练后的 UMAP 可视化
fig2, ax2 = plt.subplots(figsize=(12, 10))
scatter2 = ax2.scatter(post_train_umap_features_jittered[:, 0], post_train_umap_features_jittered[:, 1],
                       c=post_train_labels, cmap='tab20', s=10, alpha=0.3)
ax2.set_title("UMAP After Training (ResNet50)", fontsize=16)
ax2.set_xlabel("UMAP Component 1", fontsize=14)
ax2.set_ylabel("UMAP Component 2", fontsize=14)
ax2.set_xticks([])  # Hide x-axis ticks and labels
ax2.set_yticks([])  # Hide y-axis ticks and labels
# 添加颜色条
cbar2 = fig2.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar2.set_ticks(range(num_classes))
cbar2.set_label('Classes', fontsize=14)
cbar2.set_ticklabels(label_encoder.classes_)
cbar2.ax.tick_params(rotation=45, labelsize=10)

plt.savefig("ResNet50_umap_after_training_jittered.png", bbox_inches='tight', dpi=1200)
plt.show()