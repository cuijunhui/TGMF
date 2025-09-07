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
                nn.Conv1d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
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
    def __init__(self, blocks=[3, 4, 23, 3], num_classes=6, expansion=4):
        super(ResNet50, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=1, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 2048, L']

# ---------------------- 多模态Transformer分类器 ----------------------
class MultimodalTransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, num_layers=2, dropout=0.2):
        super(MultimodalTransformerClassifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features=False):
        x = x.unsqueeze(0)  # [1, B, input_dim]
        x = self.transformer(x)
        x = x.squeeze(0)    # [B, input_dim]
        if return_features:
            return x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

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
    "Almandine": "Raman spectrum peaks are located at '355','560','917',"
                 "The peak at 355 is formed by the stretching of the Si-O bond,"
                 "the peak at 560 is formed by the bending of the Si-O bond,"
                 "the peak at 917 is formed by the overall rotation of the SiO4.",
    "Andradite": "Raman spectrum peaks are located at '344','362','514','873',"
                 "The peak at 344,362 is formed by the stretching of the Si-O bond,"
                 "the peak at 514 is formed by the bending of the Si-O bond,"
                 "the peak at 873 is formed by the overall rotation of the SiO4.",
    "Beryl": "Raman spectrum peaks are located at '320','396','530','683','1014','1070',"
             "The peaks at 320,396 are formed by the bending of Al-O bonds, "
             "the peak at 530 is formed by the bending of O-Be-O bonds, "
             "the peak at 683 is formed by the bending of Si-O-Si bonds,"
             "the peaks at 1014,1070 are formed by the stretching of Si-O bonds.",
    "Calcite": "Raman spectrum peaks are located at '155','280','437','713','749','1084',"
               "The peak at 155 is formed by the translation of the Ca2+ lattice, "
               "the peak at 280 is formed by the swing of the [CO3]2- lattice, "
               "the peak at 437 is formed by the antisymmetric stretching vibration of the [CO3]2- group, "
               "the peak at 713 is formed by the in-plane bending vibration of the [CO3]2- group, "
               "the peak at 749 is formed by the out-of-plane bending vibration of the [CO3]2- group, "
               "the peak at 1084 is formed by the symmetric stretching vibration of the [CO3]2- group.",
    "Diamond": "Raman spectrum peak is located at '1330',"
               "The peak at 1330 is formed by the interaction of c=c bonds.",
    "Diopside": "Raman spectrum peaks are located at '321','386','663','1009'"
                "The peak at 663 is formed by Si-O-Si bending vibration, "
                "the peak at 1009 is formed by Si-O-Si symmetric stretching vibration, "
                "the peak at 321 is formed by M-O bending vibration, "
                "the peak at 386 is formed by M-O deformation vibration.",
    "Elbaite": "Raman spectrum peaks are located at '222','248','374','711','752','1063',"
               "The peaks at 222, 248, and 374 are formed by [Si6O18]12-deformation stretching vibration, "
               "the peaks at 711 and 752 are formed by [BO3]3- deformation stretching vibration, "
               "the peak at 1063 is formed by Si-O stretching vibration.",
    "Epidote": "Raman spectrum peaks are located at '229','302','312','316','384','404','428','437','579','584','589','673','683','858','930','948','953','1034',"
               "The peaks at 229, 302, 312, and 316 are formed by lattice vibration, "
               "the peaks at 384, 404, and 428 are formed by the stretching vibration of Fe-O bonds, "
               "the peaks at 437, 579, 584, 589, 673, 683, 858, 930, 935, 948, 953, and 1034 are formed by the symmetric stretching vibration of Si-O.",
    "Forsterite": "Raman spectrum peaks are located at '438','619','827','856','917','965',"
                  "The peak at 438 is formed by the antisymmetric bending vibration of Mg-O, "
                  "the peak at 619 is formed by the antisymmetric deformation vibration of O-Si-O, "
                  "the peaks at 827 and 856 are formed by the symmetric stretching vibration of [SiO4],"
                  "the peaks at 917 and 965 are formed by the antisymmetric stretching vibration of [SiO4].",
    "Grossular": "Raman spectrum peaks are located at '181','246','280','373','416','549','827','880',"
                 "The peak at 181 is formed by the translational vibration of Ca2+, "
                 "the peaks at 246 and 280 are formed by the translational vibration of [SiO4] tetrahedron, "
                 "the peak at 373 is formed by the rotational vibration of [SiO4] tetrahedron, "
                 "the peaks at 416 and 549 are formed by the bending vibration of [SiO4] tetrahedron, "
                 "the peaks at 827 and 880 are formed by the stretching vibration of [SiO4] tetrahedron.",
    "Muscovite": "Raman spectrum peaks are located at '203','432','709',"
                 "The peaks at 203 and 432 are formed by lattice vibration and cation exchange,"
                 "The peak at 709 is formed by the stretching and bending vibration of the Si-Obr-Si tetrahedron and the stretching vibration of the Al-O bridge oxygen bond.",
    "Pyrope": "Raman spectrum peaks are located at '355','552','912',"
              "The peak at 355 is formed by the rotation of the (SiO4) 4-group, "
              "the peak at 552 is formed by Si-O bending, "
              "the peak at 912 is formed by the stretching of Si-O.",
    "Fluorapatite": "Raman spectrum peak is located at '964',"
                   "The peak at 964 is formed by (PO4) vibration.",
    "Marialite": "Raman spectrum peaks are located at '265','459','538','775',"
                 "The peak at 256 is formed by the vibration of M-O,"
                 "The peak at 775 is formed by the symmetrical stretching vibration between Al-O.",
    "Enstatite": "Raman spectrum peaks are located at '1026','1010','931','682','660','576','404','342','238',"
                 "The peak at 1026 is formed by the asymmetric stretching of Si-O, "
                 "the peak at 1010 is formed by the symmetric stretching of Si-O, "
                 "the peak at 931 is formed by the symmetric stretching of Si-O, "
                 "the peak at 682 is formed by the symmetric stretching of Si-O-Si, "
                 "the peak at 660 is formed by the symmetric bending of Si-O-Si, "
                 "the peak at 576 is formed by the bending vibration of O-Si-O, "
                 "the peak at 404 is formed by the deformation vibration of M-O, "
                 "the peak at 342 is formed by the bending vibration of M-O, "
                 "the peak at 238 is formed by the bending vibration of M-O.",
    "Dolomite": "Raman spectrum peaks are located at '200','282','725','1090',"
                "The peaks at 200 and 282 are formed by the lattice vibrations of carbonate minerals, "
                "the peak at 725 is formed by the symmetric stretching of C-O, "
                "the peak at 1090 is formed by the bending vibration of C-O.",
    "Anorthite": "Raman spectrum peaks are located at '102','186','290','489','516','572','800',"
                 "The peaks at 102, 186, and 290 are formed by the vibrations between metal cations (M) and oxygen, "
                 "the peaks at 489, 516, and 572 are formed by the bending vibrations of O-Si(Al)-O and the asymmetric stretching of Si-Obr-Si(Al), "
                 "the peak at 800 is formed by the vibration of the SiO4 tetrahedron.",
    "Wendwilsonite": "Raman spectrum peaks are located at '180','350','370','450','820'.",
    "Quartz": "Raman spectrum peaks are located at '130','206','465',"
              "The peak at 130 is formed by lattice bending, "
              "the peak at 206 is formed by the stretching vibration of Si-O-Si bonds, "
              "the peak at 465 is formed by the vibration of Si-O-Si bonds.",
    "Albite": "Raman spectrum peaks are located at '163','478','506',"
              "The peak at 163 is formed by lattice vibrations, "
              "the peak at 478 is formed by the stretching vibration of Si-O bonds, "
              "the peak at 506 is formed by the deformation vibration of Si-O-Al bonds."
}
desc_list = [descriptions.get(name, "") for name in names]
X_train, X_test, y_train, y_test, desc_train, desc_test = train_test_split(
    spectra, labels, desc_list, test_size=0.3, random_state=42, stratify=labels
)

# ---------------------- 加载文本模型 ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
text_model = AutoModel.from_pretrained(model_name).to(device)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
text_model.resize_token_embeddings(len(tokenizer))

# ---------------------- 定义多模态模型 ----------------------
num_class = len(label_encoder.classes_)
dropout = 0.3
resnet_extractor = ResNet50(num_classes=-1).to(device)

def extract_resnet_features(x):
    out = resnet_extractor(x)  # [B, 2048, L']
    return torch.mean(out, dim=2)  # [B, 2048]

resnet_proj = nn.Linear(2048, 512).to(device)
combined_feature_dim = 512 + text_model.config.hidden_size  # 512 + 768 = 1280

classifier = MultimodalTransformerClassifier(
    input_dim=combined_feature_dim,
    hidden_dim=32,
    num_classes=num_class,
    num_heads=4,
    num_layers=2,
    dropout=dropout
).to(device)

# ---------------------- 特征提取函数 ----------------------
def extract_features(resnet_extractor, resnet_proj, text_model, tokenizer, spectra, descriptions, device, batch_size=32, labels=None):
    resnet_extractor.eval()
    text_model.eval()
    classifier.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch_spectra = spectra[i:i + batch_size]
            batch_descriptions = descriptions[i:i + batch_size]
            batch_labels = labels[i:i + batch_size] if labels is not None else None
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).to(device).unsqueeze(1)
            resnet_feats = extract_resnet_features(spectra_tensor)  # [B, 2048]
            raman_features = resnet_proj(resnet_feats)  # [B, 512]
            inputs = tokenizer(batch_descriptions, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            text_features = text_model(**inputs).last_hidden_state[:, 0, :]  # [B, 768]
            combined_features = torch.cat((raman_features, text_features), dim=1)  # [B, 1280]
            features = classifier(combined_features, return_features=True)  # [B, 1280]
            features_list.append(features.cpu().numpy())
            if batch_labels is not None:
                labels_list.extend(batch_labels)

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list) if labels_list else None

    expected_num_samples = len(spectra)
    if features.shape[0] != expected_num_samples:
        raise ValueError(f"Number of extracted features ({features.shape[0]}) does not match input samples ({expected_num_samples})")
    if labels is not None and len(labels) != expected_num_samples:
        raise ValueError(f"Number of labels ({len(labels)}) does not match input samples ({expected_num_samples})")

    return features, labels

# ---------------------- 提取训练前特征 ----------------------
pre_train_features, pre_train_labels = extract_features(
    resnet_extractor, resnet_proj, text_model, tokenizer, X_test, desc_test, device, batch_size=32, labels=y_test
)
print(f"Pre-train features shape: {pre_train_features.shape}")
print(f"Pre-train labels shape: {pre_train_labels.shape}")

# 应用 UMAP（训练前）
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
pre_train_umap_features = umap_reducer.fit_transform(pre_train_features)
print(f"Pre-train UMAP features shape: {pre_train_umap_features.shape}")

# ---------------------- 训练设置 ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(classifier.parameters()) + list(resnet_extractor.parameters()) + list(resnet_proj.parameters()) + list(text_model.parameters()),
    lr=0.00005, weight_decay=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
batch_size = 32
epochs = 10
max_grad_norm = 2.0

# ---------------------- 训练循环 ----------------------
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    classifier.train()
    resnet_extractor.train()
    text_model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for i in range(0, len(X_train), batch_size):
        batch_spectra = X_train[i:i+batch_size]
        batch_labels = y_train_tensor[i:i+batch_size]
        batch_descriptions = desc_train[i:i+batch_size]
        spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).unsqueeze(1).to(device)
        resnet_feats = extract_resnet_features(spectra_tensor)  # [B, 2048]
        raman_features = resnet_proj(resnet_feats)  # [B, 512]
        inputs = tokenizer(batch_descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
        text_features = text_model(**inputs).last_hidden_state[:, 0, :]  # [B, 768]
        combined_features = torch.cat((raman_features, text_features), dim=1)
        optimizer.zero_grad()
        outputs = classifier(combined_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(classifier.parameters()) + list(resnet_extractor.parameters()) + list(resnet_proj.parameters()) + list(text_model.parameters()),
            max_grad_norm
        )
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    train_acc = accuracy_score(all_labels, all_preds)
    train_losses.append(total_loss / len(X_train))
    train_accuracies.append(train_acc)

    classifier.eval()
    resnet_extractor.eval()
    text_model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_spectra = X_test[i:i+batch_size]
            batch_labels = y_test_tensor[i:i+batch_size]
            batch_descriptions = desc_test[i:i+batch_size]
            spectra_tensor = torch.tensor(batch_spectra, dtype=torch.float32).unsqueeze(1).to(device)
            resnet_feats = extract_resnet_features(spectra_tensor)
            raman_features = resnet_proj(resnet_feats)
            inputs = tokenizer(batch_descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
            text_features = text_model(**inputs).last_hidden_state[:, 0, :]
            combined_features = torch.cat((raman_features, text_features), dim=1)
            outputs = classifier(combined_features)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
    test_acc = accuracy_score(test_labels, test_preds)
    test_losses.append(test_loss / len(X_test))
    test_accuracies.append(test_acc)
    # scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# ---------------------- 提取训练后特征 ----------------------
post_train_features, post_train_labels = extract_features(
    resnet_extractor, resnet_proj, text_model, tokenizer, X_test, desc_test, device, batch_size=32, labels=y_test
)
print(f"Post-train features shape: {post_train_features.shape}")
print(f"Post-train labels shape: {post_train_labels.shape}")

# 应用 UMAP（训练后）
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
post_train_umap_features = umap_reducer.fit_transform(post_train_features)
print(f"Post-train UMAP features shape: {post_train_umap_features.shape}")

# ---------------------- 绘制混淆矩阵 ----------------------
class_names = label_encoder.classes_
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (ResNet50+Bert)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix_resnet50_bert.png", bbox_inches='tight')
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
ax1.set_title("UMAP Before Training (ResNet50+Bert)", fontsize=16)
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

plt.savefig("ResNet50_bert_umap_before_training_jittered.png", bbox_inches='tight', dpi=1200)
plt.show()

# 绘制训练后的 UMAP 可视化
fig2, ax2 = plt.subplots(figsize=(10, 8))
scatter2 = ax2.scatter(post_train_umap_features_jittered[:, 0], post_train_umap_features_jittered[:, 1],
                       c=post_train_labels, cmap='tab20', s=10, alpha=0.3)
ax2.set_title("UMAP After Training (ResNet50+Bert)", fontsize=16)
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

plt.savefig("ResNet50_bert_umap_after_training_jittered.png", bbox_inches='tight', dpi=1200)
plt.show()