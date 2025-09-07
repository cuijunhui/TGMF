# Breaking Modal Barriers: Text-Prompt Guided Raman Spectroscopy-Based Mineral Identification

# Overview
This project introduces a Text-Prompt-Guided Multimodal Fusion (TGMF) model designed for cross-modal modeling between Raman spectroscopy data and mineral structural texts. The proposed multimodal architecture integrates four core components:

![model](https://github.com/cuijunhui/TGMF/blob/main/model.png)

1.**Spectral Encoder**: Extracts discriminative features from the Raman spectral data.

2.**Text Encoder**: Utilizes a BERT-based model to generate semantic embeddings from textual descriptions of mineral structures.

3.**Dual Cross-Attention Module**: Enables deep interaction between the spectral and textual modalities by leveraging semantic clues to guide attention, enhancing cross-modal alignment.

4.**Fusion-based Classification Module**: Combines attended features to perform accurate mineral identification.
# Installation
To set up the project environment:

1.Clone the repository

```git clone https://github.com/cuijunhui/TGMF.git```

2.Install required Python packages:

```conda install -r requirements.txt```

# Data

[**RRUFF**](https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip)


