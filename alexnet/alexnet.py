import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import umap

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

# ---------------------- Spectral FC Encoder Module (AlexNet) ----------------------
class AlexNet(nn.Module):
    def __init__(self, num_classes, input_size=1700):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, return_features=False, return_fc1=False):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        if return_features:
            return features
        if return_fc1:
            x = self.classifier[0:3](features)
            return x
        x = self.classifier(features)
        return x

# ---------------------- 参数初始化 ----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(m.bias)

# ---------------------- 提取特征函数 ----------------------
def extract_features(model, spectra, device, labels=None, batch_size=128, return_fc1=False):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch_spectra = spectra[i:i + batch_size]
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device).unsqueeze(1)
            features = model(spectra_tensor, return_features=not return_fc1, return_fc1=return_fc1)
            features_list.append(features.cpu().numpy())
            if labels is not None:
                batch_labels = labels[i:i + batch_size]
                labels_list.extend(batch_labels)

    features = np.concatenate(features_list, axis=0)
    labels_out = np.array(labels_list) if labels_list else None

    expected_num_samples = len(spectra)
    if features.shape[0] != expected_num_samples:
        raise ValueError(
            f"Number of extracted features ({features.shape[0]}) does not match input samples ({expected_num_samples})")
    if labels_out is not None and len(labels_out) != expected_num_samples:
        raise ValueError(f"Number of labels ({len(labels_out)}) does not match input samples ({expected_num_samples})")

    return features, labels_out

# ---------------------- 计算类内和类间距离以评估特征质量 ----------------------
def compute_class_distances(features, labels):
    num_classes = len(np.unique(labels))
    class_means = np.zeros((num_classes, features.shape[1]))
    intra_class_distances = []
    inter_class_distances = []

    for i in range(num_classes):
        class_features = features[labels == i]
        if len(class_features) == 0:
            intra_class_distances.append(0)
            continue
        class_mean = np.mean(class_features, axis=0)
        class_means[i] = class_mean
        intra_dist = np.mean([np.linalg.norm(feat - class_mean) for feat in class_features])
        intra_class_distances.append(intra_dist)

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if len(features[labels == i]) == 0 or len(features[labels == j]) == 0:
                continue
            inter_dist = np.linalg.norm(class_means[i] - class_means[j])
            inter_class_distances.append(inter_dist)

    avg_intra_class_dist = np.mean(intra_class_distances)
    avg_inter_class_dist = np.mean(inter_class_distances) if inter_class_distances else 0

    return avg_intra_class_dist, avg_inter_class_dist

# ---------------------- 计算准确率、灵敏度、特异性 ----------------------
def compute_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
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

# 检查类别分布
print(pd.Series(labels).value_counts())

# ---------------------- 5-Fold 交叉验证设置 ----------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
num_classes = len(label_encoder.classes_)

# 存储交叉验证结果
fold_accuracies = []
fold_sensitivities = []
fold_specificities = []
fold_confusion_matrices = []
fold_train_losses = []
fold_test_losses = []
fold_train_accuracies = []
fold_test_accuracies = []
best_accuracy = 0
best_model_state = None
best_post_train_features = None
best_post_train_labels = None
best_fold = -1

# 交叉验证循环
for fold, (train_idx, test_idx) in enumerate(kf.split(spectra, labels)):
    print(f"\nFold {fold + 1}/{n_splits}")
    X_train, X_test = spectra[train_idx], spectra[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 定义模型
    model = AlexNet(num_classes=num_classes, input_size=1700).to(device)
    model.apply(init_weights)

    # 训练设置
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    epochs = 100
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    max_grad_norm = 5.0
    batch_size = 128

    # 训练循环
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

        # scheduler.step(test_loss)

        print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存折的训练和测试结果
    fold_train_losses.append(train_losses)
    fold_test_losses.append(test_losses)
    fold_train_accuracies.append(train_accuracies)
    fold_test_accuracies.append(test_accuracies)

    # 计算混淆矩阵和指标
    cm = confusion_matrix(test_labels, test_preds, labels=np.arange(num_classes))
    fold_confusion_matrices.append(cm)
    accuracy, avg_sensitivity, avg_specificity, sensitivity, specificity = compute_metrics(test_labels, test_preds, num_classes)
    fold_accuracies.append(accuracy)
    fold_sensitivities.append(sensitivity)  # Store per-class sensitivities
    fold_specificities.append(specificity)  # Store per-class specificities

    # 提取训练后特征
    post_train_features, post_train_labels = extract_features(model, X_test, device, labels=y_test, batch_size=batch_size, return_fc1=True)

    # 保存最佳折的模型和特征
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model.state_dict()
        best_post_train_features = post_train_features
        best_post_train_labels = post_train_labels
        best_fold = fold + 1

# ---------------------- 计算交叉验证的平均值和标准差 ----------------------
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
mean_sensitivity = np.mean([np.mean(sens) for sens in fold_sensitivities])
std_sensitivity = np.std([np.mean(sens) for sens in fold_sensitivities])
mean_specificity = np.mean([np.mean(spec) for spec in fold_specificities])
std_specificity = np.std([np.mean(spec) for spec in fold_specificities])

print("\nCross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean Sensitivity: {mean_sensitivity:.4f} ± {std_sensitivity:.4f}")
print(f"Mean Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}")

# Debug: Print fold_sensitivities to check structure
print("\nDebug: Structure of fold_sensitivities:")
for f in range(n_splits):
    print(f"Fold {f+1}: {fold_sensitivities[f]}")

# 输出每个类的灵敏度和特异性
print("\nPer-class Sensitivity (Recall) Across Folds:")
for i in range(num_classes):
    class_sensitivities = []
    for f in range(n_splits):
        if len(fold_sensitivities[f]) > i:  # Check if index i is valid
            class_sensitivities.append(fold_sensitivities[f][i])
        else:
            class_sensitivities.append(0)  # Handle missing class
    class_sensitivity = np.mean(class_sensitivities)
    class_std_sensitivity = np.std(class_sensitivities)
    print(f"Class {label_encoder.classes_[i]}: {class_sensitivity:.4f} ± {class_std_sensitivity:.4f}")

print("\nPer-class Specificity Across Folds:")
for i in range(num_classes):
    class_specificities = []
    for f in range(n_splits):
        if len(fold_specificities[f]) > i:  # Check if index i is valid
            class_specificities.append(fold_specificities[f][i])
        else:
            class_specificities.append(0)  # Handle missing class
    class_specificity = np.mean(class_specificities)
    class_std_specificity = np.std(class_specificities)
    print(f"Class {label_encoder.classes_[i]}: {class_specificity:.4f} ± {class_std_specificity:.4f}")

# ---------------------- 计算平均混淆矩阵 ----------------------
avg_cm = np.mean(fold_confusion_matrices, axis=0)
avg_cm = np.round(avg_cm).astype(int)  # 四舍五入到整数

# 绘制平均混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(avg_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Average Confusion Matrix (AlexNet)",fontsize=16)
plt.xlabel("Predicted Labels",fontsize=14)
plt.ylabel("True Labels",fontsize=14)
plt.savefig("average_confusion_matrix_alexnet_5fold.png", bbox_inches='tight',dpi=1200)
plt.show()

# ---------------------- 最佳折的特征提取与可视化 ----------------------
# Preprocess best fold features with PCA
pca = PCA(n_components=336)
best_post_train_features_pca = pca.fit_transform(best_post_train_features)
print(f"Best fold (Fold {best_fold}) PCA features shape: {best_post_train_features_pca.shape}")
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# 计算类内和类间距离
pre_intra_dist, pre_inter_dist = compute_class_distances(best_post_train_features, best_post_train_labels)
post_intra_dist, post_inter_dist = compute_class_distances(best_post_train_features, best_post_train_labels)
print(f"Best fold (Fold {best_fold}) Pre-train: Avg intra-class distance: {pre_intra_dist:.4f}, Avg inter-class distance: {pre_inter_dist:.4f}")
print(f"Best fold (Fold {best_fold}) Post-train: Avg intra-class distance: {post_intra_dist:.4f}, Avg inter-class distance: {post_inter_dist:.4f}")

# 应用 UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.05, metric='euclidean')
best_post_train_umap_features = umap_reducer.fit_transform(best_post_train_features_pca, y=best_post_train_labels)
print(f"Best fold UMAP features shape: {best_post_train_umap_features.shape}")

# 绘制最佳折的 UMAP（带抖动）
fig, ax = plt.subplots(figsize=(12, 10))
jitter = 0.2
best_post_train_umap_features_jittered = best_post_train_umap_features + np.random.normal(0, jitter, best_post_train_umap_features.shape)
scatter = ax.scatter(best_post_train_umap_features_jittered[:, 0], best_post_train_umap_features_jittered[:, 1],
                    c=best_post_train_labels, cmap='tab20', s=20, alpha=0.5)
ax.set_title(f"UMAP After Training (alexnet)", fontsize=16)
ax.set_xlabel("UMAP Component 1", fontsize=14)
ax.set_ylabel("UMAP Component 2", fontsize=14)
ax.set_xticks([])  # Hide x-axis ticks and labels
ax.set_yticks([])  # Hide y-axis ticks and labels
cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar.set_ticks(range(num_classes))
cbar.set_label('Classes', fontsize=14)
cbar.set_ticklabels(label_encoder.classes_)
cbar.ax.tick_params(rotation=45, labelsize=10)
plt.savefig(f"AlexNet_umap_best_fold_{best_fold}.png", bbox_inches='tight', dpi=1200)
plt.show()

# ---------------------- 最佳折的子采样 UMAP 可视化 ----------------------
subsample_indices = []
samples_per_class = 10
for i in range(num_classes):
    class_indices = np.where(best_post_train_labels == i)[0]
    subsample_indices.extend(class_indices[:samples_per_class])
subsample_indices = np.array(subsample_indices)
sub_best_post_train_umap_features = best_post_train_umap_features[subsample_indices]
sub_best_post_train_labels = best_post_train_labels[subsample_indices]

fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(sub_best_post_train_umap_features[:, 0], sub_best_post_train_umap_features[:, 1],
                     c=sub_best_post_train_labels, cmap='tab20', s=100, alpha=0.8)
ax.set_title(f"UMAP After Training (AlexNet, Subsampled, Best Fold {best_fold})")
ax.set_xlabel("UMAP Component 1")
ax.set_ylabel("UMAP Component 2")
cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar.set_ticks(range(num_classes))
cbar.set_label('Classes', fontsize=12)
cbar.set_ticklabels(label_encoder.classes_)
cbar.ax.tick_params(rotation=45, labelsize=10)
plt.savefig(f"AlexNet_umap_subsampled_best_fold_{best_fold}.png", bbox_inches='tight')
plt.show()

# ---------------------- 结合合成文本特征的 UMAP 可视化（最佳折） ----------------------
one_hot = OneHotEncoder(sparse_output=False)
text_like_features = one_hot.fit_transform(best_post_train_labels.reshape(-1, 1))
best_post_train_combined = np.concatenate([best_post_train_features_pca, text_like_features], axis=1)
print(f"Best fold combined features shape: {best_post_train_combined.shape}")

umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.05)
best_post_train_umap_combined = umap_reducer.fit_transform(best_post_train_combined)
print(f"Best fold UMAP combined features shape: {best_post_train_umap_combined.shape}")

fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(best_post_train_umap_combined[:, 0], best_post_train_umap_combined[:, 1],
                     c=best_post_train_labels, cmap='tab20', s=20, alpha=0.5)
ax.set_title(f"UMAP After Training (AlexNet + Synthetic Text Features, Best Fold {best_fold})")
ax.set_xlabel("UMAP Component 1")
ax.set_ylabel("UMAP Component 2")
cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar.set_ticks(range(num_classes))
cbar.set_label('Classes', fontsize=20)
cbar.set_ticklabels(label_encoder.classes_)
cbar.ax.tick_params(rotation=45, labelsize=20)
plt.savefig(f"AlexNet_umap_with_synthetic_text_best_fold_{best_fold}.png", bbox_inches='tight',dpi=1200)
plt.show()

# ---------------------- 绘制交叉验证的损失曲线 ----------------------
plt.figure(figsize=(10, 5))
for fold in range(n_splits):
    plt.plot(fold_train_losses[fold], label=f'Fold {fold + 1} Train Loss', alpha=0.5)
    plt.plot(fold_test_losses[fold], label=f'Fold {fold + 1} Test Loss', linestyle='--', alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves Across Folds')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('loss_curves_5fold.png')
plt.show()