# Setup Guide - Data Driven Influencer Marketing

Complete step-by-step installation and configuration guide.

### 1. Install Dependencies

```bash

# Install requirements
pip install -r requirements.txt

# If using GPU (CUDA 12.1)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric
```





```
influencer-ranking/
├── venv/                           # Virtual environment
├── actual_dataset/
│   ├── images/                     # ~50 GB of influencer images
│   └── info/                       # JSON post metadata
├── gen_dataset/
│   ├── influencers_17.csv
│   └── JSON-image_17.csv
├── dataset/
│   ├── images/                     # Organized images
│   ├── info/                       # Organized metadata
│   ├── influencers.csv
│   └── JSON-image_17.csv
├── image_embed/
│   ├── image_feature_extractor.keras    # Downloaded model
│   ├── compressPreprocess.py
│   ├── extract_image_features.py
│   ├── compressedPreprocessedImages/
│   └── image_features/
├── text_embed/
│   ├── pipeline.py
│   ├── embed.py
│   └── processed_posts.csv
├── combined_features/
│   ├── combined_feature_vectors.py
│   └── combined_feature_vectors/
├── classifier/
│   ├── influencer_profiler_best.keras   # Downloaded model
│   ├── run_classificatoin_for_folder.py
│   └── predictions.csv
├── ranking/
│   ├── build_enhanced_graphs_v3_fixed.py
│   ├── parse_profiles.py
│   ├── v8-final.ipynb
│   ├── predict_rankings.py
│   ├── graphs_enhanced_v3/              # Downloaded graphs
│   └── saved_models_v8_final/           # Downloaded model
├── interface/
│   ├── frontend/
│   │   └── working_frontend.py
│   └── backend/
│       └── working_reciever.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── SETUP.md                        # This file
├── ARCHITECTURE.md
├── API.md
└── TRAINING.md
```


## 💾 Storage Requirements

| Component | Size | Type |
|---|---|---|
| Instagram_posts | 40-200 GB | Essential |
| Metadata | 100 MB | Essential |
| Models | 20 MB | Essential |
| Graphs | 2-3 GB | Essential |
| Embeddings | 5-10 GB | Generated |
| Features | 2-5 GB | Generated |
| **Total** | **~60-230 GB** | - |
