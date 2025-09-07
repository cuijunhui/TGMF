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
from scipy.interpolate import interp1d


# ---------------------- Data Augmentation Functions ----------------------
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


def balance_dataset(spectra, labels, wavenumber_range=(100, 1799), shift_range=(-0.5, 0.5)):
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    print(f"Max sample count: {max_count}")

    augmented_spectra = []
    augmented_labels = []

    for label in unique_labels:
        mask = labels == label
        class_spectra = spectra[mask]
        class_count = len(class_spectra)

        if class_count < max_count:
            num_to_augment = max_count - class_count
            indices = np.random.choice(class_count, num_to_augment, replace=True)
            spectra_to_augment = class_spectra[indices]
            augmented = augment_raman_shift(spectra_to_augment, wavenumber_range, shift_range)
            augmented_spectra.append(class_spectra)
            augmented_spectra.append(augmented)
            augmented_labels.extend([label] * class_count)
            augmented_labels.extend([label] * num_to_augment)
        else:
            augmented_spectra.append(class_spectra)
            augmented_labels.extend([label] * class_count)

    augmented_spectra = np.concatenate(augmented_spectra, axis=0)
    augmented_labels = np.array(augmented_labels)
    return augmented_spectra, augmented_labels


# ---------------------- VGG Network Definition ----------------------
def make_vgg_layer(in_channels, out_channels, num_blocks, dilation=1, ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers


class VGG(nn.Module):
    def __init__(self, depth=11):
        super(VGG, self).__init__()
        arch_settings = {11: (1, 1, 2, 2, 2)}
        stage_blocks = arch_settings[depth]
        in_channels = 1
        layers = []
        for i, num_blocks in enumerate(stage_blocks):
            out_channels = 64 * 2 ** i if i < 4 else 512
            layers += make_vgg_layer(in_channels, out_channels, num_blocks)
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x  # [B, 512, 1]


# ---------------------- Prediction ----------------------
class FCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # [B, 1, d_model] -> [B, d_model]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # [B, num_classes]


# ---------------------- Multimodal Classifier (Spectra Only) ----------------------
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, d_model=256, hidden_dim=128, dropout=0.5):
        super(MultimodalClassifier, self).__init__()
        self.raman_extractor = VGG(depth=11)
        self.raman_proj = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = FCClassifier(d_model, hidden_dim, num_classes)

    def forward(self, spectra, return_features=False):
        # Extract Raman features
        x_spec = spectra.unsqueeze(1)  # [B, 1, L]
        raman_out = self.raman_extractor(x_spec)  # [B, 512, 1]
        raman_seq = raman_out.permute(0, 2, 1)  # [B, 1, 512]
        raman_seq = self.raman_proj(raman_seq)  # [B, 1, d_model]

        # Adjust sequence lengths
        fixed_len = 1
        if raman_seq.shape[1] != fixed_len:
            raman_seq = F.adaptive_avg_pool1d(raman_seq.permute(0, 2, 1), fixed_len).permute(0, 2, 1)

        raman_seq = self.dropout(raman_seq)

        # Classification
        logits = self.classifier(raman_seq)

        if return_features:
            return logits, raman_seq
        return logits


# ---------------------- Parameter Initialization ----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------- Data Loading and Preprocessing ----------------------
file_path = "../excellent_unoriented_data_top20_moveline.csv"
data = pd.read_csv(file_path)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["Name"].values)
spectra = data.iloc[:, 1:].values
scaler = StandardScaler()
spectra = scaler.fit_transform(spectra)

# Data augmentation
spectra, labels = balance_dataset(spectra, labels)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    spectra, labels, test_size=0.3, random_state=7941123, stratify=labels
)

# Check class distribution
print(pd.Series(labels).value_counts())
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Unique y_train values: {np.unique(y_train)}")
print(f"Unique y_test values: {np.unique(y_test)}")

# ---------------------- Define Model ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
num_class = len(label_encoder.classes_)
dropout = 0.5
model = MultimodalClassifier(
    num_classes=num_class,
    d_model=256,
    hidden_dim=128,
    dropout=dropout
).to(device)
model.apply(init_weights)


# ---------------------- Extract Pre-training Features ----------------------
def extract_features(model, spectra, device, batch_size=128, labels=None):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch_spectra = spectra[i:i + batch_size]
            batch_labels = labels[i:i + batch_size] if labels is not None else None
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device)

            _, fused_features = model(spectra_tensor, return_features=True)
            fused_features = fused_features.mean(dim=1)  # [B, d_model]
            features_list.append(fused_features.cpu().numpy())
            if batch_labels is not None:
                labels_list.extend(batch_labels)

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list) if labels is not None else None

    expected_num_samples = len(spectra)
    if features.shape[0] != expected_num_samples:
        raise ValueError(
            f"Number of extracted features ({features.shape[0]}) does not match input samples ({expected_num_samples})")
    if labels is not None and len(labels) != expected_num_samples:
        raise ValueError(f"Number of labels ({len(labels)}) does not match input samples ({expected_num_samples})")

    return features, labels


# Extract pre-training test set features
pre_train_features, pre_train_labels = extract_features(model, X_test, device, batch_size=128, labels=y_test)
print(f"Pre-train features shape: {pre_train_features.shape}")
print(f"Pre-train labels shape: {pre_train_labels.shape}")
print(f"Unique pre-train labels: {np.unique(pre_train_labels, return_counts=True)}")

# Apply UMAP (Pre-training)
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
pre_train_umap_features = umap_reducer.fit_transform(pre_train_features)
print(f"Pre-train UMAP features shape: {pre_train_umap_features.shape}")
unique_pre_train_umap_points = np.unique(pre_train_umap_features, axis=0)
print(f"Number of unique pre-train UMAP points: {unique_pre_train_umap_points.shape[0]}")

# ---------------------- Training Setup ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
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
max_grad_norm = 2.0

# Initialize test_labels and test_preds
test_preds = []
test_labels = []

# ---------------------- Training Loop ----------------------
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
        spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(spectra_tensor)
        # print(
        #     # f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Outputs shape: {outputs.shape}, Batch labels shape: {batch_labels.shape}")
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

    # Clear test_preds and test_labels for current epoch
    test_preds = []
    test_labels = []
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_spectra = X_test[i:i + batch_size]
            batch_labels = y_test_tensor[i:i + batch_size]
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device)
            outputs = model(spectra_tensor)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
    test_acc = accuracy_score(test_labels, test_preds)
    test_losses.append(test_loss / len(X_test))
    test_accuracies.append(test_acc)
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")


# ---------------------- Compute Metrics ----------------------
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


# Compute test set metrics
# print(f"Length of test_labels: {len(test_labels)}")
# print(f"Length of test_preds: {len(test_preds)}")


test_accuracy, test_sensitivity, test_specificity, per_class_sensitivity, per_class_specificity = compute_metrics(
    test_labels, test_preds, num_class
)

# Print results
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

# ---------------------- Extract Post-training Features ----------------------
post_train_features, post_train_labels = extract_features(model, X_test, device, batch_size=128, labels=y_test)
print(f"Post-train features shape: {post_train_features.shape}")
print(f"Post-train labels shape: {post_train_labels.shape}")
print(f"Unique post-train labels: {np.unique(post_train_labels, return_counts=True)}")

# Apply UMAP (Post-training)
umap_reducer = umap.UMAP(n_components=2, random_state=1234, n_neighbors=30, min_dist=0.1)
post_train_umap_features = umap_reducer.fit_transform(post_train_features)
# print(f"Post-train UMAP features shape: {post_train_umap_features.shape}")
unique_post_train_umap_points = np.unique(post_train_umap_features, axis=0)
# print(f"Number of unique post-train UMAP points: {unique_post_train_umap_points.shape[0]}")


# ---------------------- Compute Class Distances ----------------------
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

# ---------------------- Plot UMAP Visualizations (Pre-training) ----------------------
fig1 = plt.figure(figsize=(12, 10))  # Single figure for pre-training

# Add jitter
jitter = 0.0
pre_train_umap_features_jittered = pre_train_umap_features + np.random.normal(0, jitter, pre_train_umap_features.shape)

# Pre-training scatter plot
scatter1 = plt.scatter(pre_train_umap_features_jittered[:, 0], pre_train_umap_features_jittered[:, 1],
                      c=pre_train_labels, cmap='tab20', s=10, alpha=0.3)
plt.title("UMAP Before Training (VGGNet)", fontsize=16)
plt.xlabel("UMAP Component 1", fontsize=14)
plt.ylabel("UMAP Component 2", fontsize=14)
plt.xticks([])  # Hide x-axis ticks and labels
plt.yticks([])  # Hide y-axis ticks and labels

# Add colorbar
cbar = plt.colorbar(scatter1, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar.set_ticks(range(num_class))
cbar.set_label('Classes', fontsize=14)
cbar.set_ticklabels(label_encoder.classes_)
cbar.ax.tick_params(rotation=45, labelsize=10)

# Save pre-training plot
plt.savefig("VGGNet_umap_before_training_spectra_only.png", bbox_inches='tight', dpi=1200)
plt.show()
plt.close(fig1)  # Close the figure to free memory

# ---------------------- Plot UMAP Visualizations (Post-training) ----------------------
fig2 = plt.figure(figsize=(12, 10))  # Single figure for post-training

# Add jitter
post_train_umap_features_jittered = post_train_umap_features + np.random.normal(0, jitter, post_train_umap_features.shape)

# Post-training scatter plot
scatter2 = plt.scatter(post_train_umap_features_jittered[:, 0], post_train_umap_features_jittered[:, 1],
                      c=post_train_labels, cmap='tab20', s=10, alpha=0.3)
plt.title("UMAP After Training (VGGNet)", fontsize=16)
plt.xlabel("UMAP Component 1", fontsize=14)
plt.ylabel("UMAP Component 2", fontsize=14)
plt.xticks([])  # Hide x-axis ticks and labels
plt.yticks([])  # Hide y-axis ticks and labels
# Add colorbar
cbar = plt.colorbar(scatter2, orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar.set_ticks(range(num_class))
cbar.set_label('Classes', fontsize=14)
cbar.set_ticklabels(label_encoder.classes_)
cbar.ax.tick_params(rotation=45, labelsize=10)

# Save post-training plot
plt.savefig("VGGNet_umap_after_training_spectra_only.png", bbox_inches='tight', dpi=1200)
plt.show()
plt.close(fig2)  # Close the figure to free memory

# ---------------------- Subsample and Plot UMAP Visualizations ----------------------
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

# Pre-training (Subsampled)
scatter1 = ax1.scatter(sub_pre_train_umap_features[:, 0], sub_pre_train_umap_features[:, 1], c=sub_pre_train_labels,
                       cmap='tab20', s=50, alpha=0.6)
ax1.set_title("UMAP Before Training (VGGNet, Subsampled, Spectra Only)")
ax1.set_xlabel("UMAP Component 1")
ax1.set_ylabel("UMAP Component 2")

# Post-training (Subsampled)
scatter2 = ax2.scatter(sub_post_train_umap_features[:, 0], sub_post_train_umap_features[:, 1], c=sub_post_train_labels,
                       cmap='tab20', s=50, alpha=0.6)
ax2.set_title("UMAP After Training (VGGNet, Subsampled, Spectra Only)")
ax2.set_xlabel("UMAP Component 1")
ax2.set_ylabel("UMAP Component 2")

# Add colorbar
cbar = fig.colorbar(scatter2, ax=[ax1, ax2], orientation='horizontal', pad=0.15, fraction=0.05, aspect=50, shrink=0.8)
cbar.set_ticks(range(num_class))
cbar.set_label('Classes', fontsize=12)
cbar.set_ticklabels(label_encoder.classes_)
cbar.ax.tick_params(rotation=45, labelsize=10)

plt.savefig("VGGNet_umap_comparison_before_after_training_spectra_only_subsampled.png", bbox_inches='tight')
plt.show()

# Plot confusion matrix
class_names = label_encoder.classes_
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (VGGNet)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix_top20_spectra_only_VGGNet.png", bbox_inches='tight',dpi=1200)
plt.show()